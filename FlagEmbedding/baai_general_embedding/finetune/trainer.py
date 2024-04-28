from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *
from transformers.trainer_utils import is_main_process
from transformers import TrainerCallback, TrainerState, TrainerControl, IntervalStrategy
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, LinearLR, ConstantLR, ReduceLROnPlateau


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)

class MyCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_save = True
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.STEPS:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.STEPS:
            control.should_save = True

        return control

class BiTrainer(Trainer):
           
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if not is_main_process(self.args.local_rank):
            return
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir,
                                                pooling_mode=self.args.sentence_pooling_method,
                                                normlized=self.args.normlized)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class BiTrainer_with_optimizer(BiTrainer):
    def __init__(self, *args, custom_optimizer=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.custom_optimizer = custom_optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.custom_optimizer is not None:
            self.optimizer = self.custom_optimizer
        else:
            # 如果没传入，创建一个默认优化器
            print(f'warning!!!! no optimer in, use default adamw')
            self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        
        super().create_optimizer_and_scheduler(num_training_steps)

def get_wsd_scheduler(scheduler_args, optimizer, training_steps):
    W = scheduler_args.warmup_steps
    S = int(scheduler_args.stable_ratio *training_steps)
    D = training_steps-W-S
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1/W, total_iters=W)
    stable_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=S)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1/D,total_iters=D)
    
    milestones = [W, W+S]
    wsd_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler], milestones=milestones)
    
    return wsd_scheduler

class BiTrainer_with_wsd_scheduler(BiTrainer):
    def __init__(self, *args, scheduler_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler_args = scheduler_args

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # 定义优化器，假设 scheduler_args 已经包含了所有必要的参数
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.scheduler_args.lr,
                                     betas=self.scheduler_args.betas,
                                     eps=self.scheduler_args.eps,
                                     weight_decay=self.scheduler_args.weight_decay,
                                     correct_bias=self.scheduler_args.correct_bias)

        # 使用自定义的学习率调度器
        self.lr_scheduler = get_wsd_scheduler(self.scheduler_args, self.optimizer, num_training_steps)