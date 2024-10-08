import logging
import os
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import ModelArguments, DataArguments, SchedulerArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data import TrainDatasetForEmbedding, EmbedCollator, BalancedTrainDatasetForEmbedding, LlmEmbedCollator
from modeling import BiEncoderModel, my_modified_BiEncoderModel, my_modified_loss_1_BiEncoderModel,my_modified_loss_2_BiEncoderModel,Cocktail_BiEncoderModel, Bi_lmhead_EncoderModel
from trainer import BiTrainer, MyCallback, BiTrainer_with_optimizer
from load_llm_model import get_llm_model

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments,SchedulerArguments, TrainingArguments))
    model_args, data_args, schedule_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    schedule_args: SchedulerArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    logger.info('Config: %s', config)

    if model_args.use_my_modified_loss_model == '1':
        logger.info('use my modified loss model')
        model = my_modified_BiEncoderModel(model_name=model_args.model_name_or_path,
                            normlized=training_args.normlized,
                            sentence_pooling_method=training_args.sentence_pooling_method,
                            negatives_cross_device=training_args.negatives_cross_device,
                            temperature=training_args.temperature,
                            use_inbatch_neg=training_args.use_inbatch_neg,
                            )
    # elif model_args.llm_embedding_mode: # 用llm作为embedding的base模型
    #     logger.info('use llm as embedding base model')
    #     base_model = get_llm_model(model_args, training_args)
    #     model = 
    elif model_args.llm_head_embedding_mode:
        logger.info('正在使用llm + head作为embedding base模型')
        base_model = get_llm_model(model_args, training_args)
        model = Bi_llm_head_EncoderModel(model_name=model_args.model_name_or_path,
            llm_embedding_token_type='special',
            encode_head_size=128,
            normlized=training_args.normlized,
            sentence_pooling_method=training_args.sentence_pooling_method,
            negatives_cross_device=training_args.negatives_cross_device,
            temperature=training_args.temperature,
            use_inbatch_neg=training_args.use_inbatch_neg,
        )
    elif model_args.use_my_modified_loss_model == '2':
        logger.info('use my softmax modified loss model')
        model = my_modified_loss_1_BiEncoderModel(model_name=model_args.model_name_or_path,
                            normlized=training_args.normlized,
                            sentence_pooling_method=training_args.sentence_pooling_method,
                            negatives_cross_device=training_args.negatives_cross_device,
                            temperature=training_args.temperature,
                            use_inbatch_neg=training_args.use_inbatch_neg,
                            )
    elif model_args.use_balanced_dataset_and_loss == '1':
        logger.info('use my balanced dataset and loss')
        model = my_modified_loss_2_BiEncoderModel(model_name=model_args.model_name_or_path,
                            normlized=training_args.normlized,
                            sentence_pooling_method=training_args.sentence_pooling_method,
                            temperature=training_args.temperature,
                            use_inbatch_neg=training_args.use_inbatch_neg,
                            )
    elif model_args.train_the_cocktail_param == '1':
        logger.info('use cocktail weights train mode')
        model_list = model_args.cocktail_model_list.split(',')
        model = Cocktail_BiEncoderModel(model_name=model_args.model_name_or_path,
                            cocktail_model_list=model_list,
                            normlized=training_args.normlized,
                            sentence_pooling_method=training_args.sentence_pooling_method,
                            negatives_cross_device=training_args.negatives_cross_device)
        
        opti_para_list = [{'params':model.weight_list}]
        cocktail_optimizer = torch.optim.AdamW(opti_para_list)
    elif model_args.lh_head_mode:
        logger.info('use lm_head_mode train mode')
        if model_args.lm_head_freeze_base_model:
            logger.info('basemodel freezed!')
        model = Bi_lmhead_EncoderModel(model_name=model_args.model_name_or_path,
                            normlized=training_args.normlized,
                            sentence_pooling_method=training_args.sentence_pooling_method,
                            negatives_cross_device=training_args.negatives_cross_device,
                            temperature=training_args.temperature,
                            use_inbatch_neg=training_args.use_inbatch_neg,
                            freeze_base_model=model_args.lm_head_freeze_base_model,
                            )
    else:
        logger.info('use base model')
        model = BiEncoderModel(model_name=model_args.model_name_or_path,
                            normlized=training_args.normlized,
                            sentence_pooling_method=training_args.sentence_pooling_method,
                            negatives_cross_device=training_args.negatives_cross_device,
                            temperature=training_args.temperature,
                            use_inbatch_neg=training_args.use_inbatch_neg,
                            )

    training_args.lh_head_mode = model_args.lh_head_mode

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    if model_args.use_balanced_dataset_and_loss == '1':
        logger.info('dataset: BalancedTrainDatasetForEmbedding')
        train_dataset = BalancedTrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    else:
        logger.info('dataset: TrainDatasetForEmbedding（正常模式）')
        train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    # resume_from_checkpoint
    save_strategy = training_args.save_strategy
    resume_from_checkpoint_flag = os.path.exists(training_args.output_dir) and len(os.listdir(training_args.output_dir)) > 0
    if save_strategy == 'steps':
        mycallback = MyCallback()
        training_args.callbacks = [mycallback]

    if model_args.train_the_cocktail_param == '1':
        logger.info(f'using BiTrainer_with_optimizer')
        trainer = BiTrainer_with_optimizer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=EmbedCollator(
                tokenizer,
                query_max_len=data_args.query_max_len,
                passage_max_len=data_args.passage_max_len
            ),
            custom_optimizer=cocktail_optimizer,
            tokenizer=tokenizer
        )
    elif schedule_args.use_my_wsd_sceduler == '1':
        logger.info(f'using BiTrainer_with_optimizer')
        trainer = BiTrainer_with_optimizer(
            model=model,
            args=training_args,
            scheduler_args=schedule_args,
            train_dataset=train_dataset,
            data_collator=EmbedCollator(
                tokenizer,
                query_max_len=data_args.query_max_len,
                passage_max_len=data_args.passage_max_len
            ),
            tokenizer=tokenizer
        )
    elif model_args.llm_head_embedding_mode or model_args.llm_embedding_mode:
        trainer = BiTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=LlmEmbedCollator(
                tokenizer,
                query_max_len=data_args.query_max_len,
                passage_max_len=data_args.passage_max_len
            ),
            tokenizer=tokenizer
        )
    else:
        trainer = BiTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=EmbedCollator(
                tokenizer,
                query_max_len=data_args.query_max_len,
                passage_max_len=data_args.passage_max_len
            ),
            tokenizer=tokenizer
        )


    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    logger.info('fucking train')
    # trainer.train(resume_from_checkpoint=resume_from_checkpoint_flag)
    trainer.train()
    logger.info('begeinging save your fucking model!!!!shift!!!!')
    if trainer.is_world_process_zero():
        trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
