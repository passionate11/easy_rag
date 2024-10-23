import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    def __init__(self,
                 model: None,
                 tokenizer: AutoTokenizer = None,
                 train_batch_size: int = 4,
                 ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.train_batch_size = train_batch_size

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]


    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, features):
        if features is None:
            return None
        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True)
        _, max_indices = torch.max(features['labels'], dim=1)
        predict_indices = max_indices - 1
        logits = [outputs.logits[i, predict_indices[i], :] for i in range(outputs.logits.shape[0])]
        logits = torch.stack(logits, dim=0)
        scores = logits[:, self.yes_loc]
        return scores.contiguous()  # batch_size

    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None):
        ranker_logits = self.encode(pair) # (batch_size, group_size)
        if self.training:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
            loss = self.compute_loss(grouped_logits, target)
        else:
            loss = None

        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        # self.model.save_pretrained(output_dir)
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            print('here11')
            self.tokenizer.save_pretrained(output_dir)
        

    def save_pretrained(self, **kwargs):
        self.tokenizer.save_pretrained(**kwargs)
        return self.model.save_pretrained(**kwargs)

class ListwiseRerankerModel(BiEncoderModel):
    def __init__(self,
                 model: None,
                 tokenizer: AutoTokenizer = None,
                 train_batch_size: int = 4,
                 margin: float = 1.0,
                 ):
        super().__init__(model, tokenizer, train_batch_size)
        self.margin = margin



    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None):
        ranker_logits = self.encode(pair) # (batch_size * grou_size)

        if self.training:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)  # [bsz, group_size]
            pairwise_differences = grouped_logits[:, :-1] - grouped_logits[:, 1:]
            losses = torch.clamp(self.margin - pairwise_differences, min=0)
            total_loss = losses.sum(dim=1).mean()
            # target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
            # loss = self.compute_loss(grouped_logits, target)
        else:
            loss = None

        return RerankerOutput(
            loss=total_loss,
            scores=ranker_logits,
        )

        # pairwise_differences = scores[:, :-1] - scores[:, 1:]
        # # Apply the margin
        # losses = torch.clamp(self.margin - pairwise_differences, min=0)
        # # Sum up all the hinge losses for each sample and then average across the batch
        # total_loss = losses.sum(dim=1).mean()
        # return total_loss