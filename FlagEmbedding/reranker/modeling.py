import logging

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.compute_loss(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    def compute_loss(self, scores, target):
        if self.model_args.use_my_modified_loss_model == '0':
            return self.cross_entropy(scores, target)
        elif self.model_args.use_my_modified_loss_model == '1':
            # circle loss
            scores = nn.functional.softmax(scores, dim=-1)
            return self.cross_entropy(scores, target)

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)


class MixLayer(nn.Module):

    def __init__(self, layer_list, weight_list):
        super().__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.model_weight = weight_list

    def forward(self, x):
        layer_output_list = []
        for layer in self.layer_list:
            layer_output_list.append(layer(x))
        layer_output_shape = layer_output_list[0].shape
        layer_output = torch.cat(layer_output_list, 0).reshape(len(self.layer_list), *layer_output_shape)
        weight = nn.functional.softmax(self.model_weight)
        # print(f'mix weight:{weight}')
        weight = weight.reshape([-1] + [1] * len(layer_output_shape))
        return (layer_output * weight).sum(0)