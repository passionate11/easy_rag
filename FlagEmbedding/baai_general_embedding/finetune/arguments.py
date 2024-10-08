from email.policy import default
import os
from dataclasses import dataclass, field
from typing import Optional
import typing
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # my arguments
    use_my_modified_loss_model: str = field(
        default='0', metadata={"help": "whether to use my modified loss model, 0:no 1:batch no pos 2:softmax 1"})
    use_balanced_dataset_and_loss: str = field(
        default='0', metadata={"help": "whether to use balanced dataset_and_loss, 0:no 1: yes "})

    train_the_cocktail_param: str = field(
        default='0', metadata={"help": "加载多个模型,仅训练cocktail加权权重"})
    cocktail_model_list: str = field(
        default=None, metadata={"help": "cocktail时的多个模型列表"})

    lh_head_mode: Optional[bool] = field(
        default=False, metadata={"help": "是否训练一个额外的分类lm head做同等维度的映射训练"}
    )
    lm_head_freeze_base_model: Optional[bool] = field(
        default=True, metadata={"help": "lh_head_mode为True时是否冻结base model,默认冻结"}
    )
    llm_embedding_mode: Optional[bool] = field(
        default=False, metadata={"help": "是否使用llm作为embedding base模型"}
    )
    llm_head_embedding_mode: Optional[bool] = field(
        default=False, metadata={"help": "是否使用llm加一个head作为embedding base模型"}
    )
    llm_embedding_token_type: Optional[str] = field(
        default='eos', metadata={"help": "llm_embedding_mode为True时, 使用什么作为embedding的向量, 默认为eos"}
    )
    

@dataclass
class SchedulerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    use_my_wsd_sceduler: str = field(
        default='0', metadata={"help": "whether to use my wsd scheduler, 0:no 1: yes "})
    # for adamw
    stable_ratio: float = field(
        default=0.1, metadata={"help": "the stable ratio for wsd scheduler"}
    )

    
@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})
