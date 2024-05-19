import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model: None,
                 use_model_type: str = None,
                #  model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = model
        self.use_model_type = use_model_type
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        print(f"psg_out in encode:\n{psg_out}")
        print(f"psg_out in encode shape:\n{psg_out.last_hidden_state.shape}")
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        print(f"p_reps in encode after sentence embedding:\n{p_reps}")
        print(f"p_reps in encode after sentence embedding shape:\n{p_reps.shape}")
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def inference_qp(self, qp_data):
        model = self.model.model
        embeddings = [] # 存储所有数据的embedding
        cur_in_ids = {
                        'input_ids': qp_data['input_ids'],
                        'attention_mask':qp_data['attention_mask']
                        }
        
        results = model(**cur_in_ids,return_dict=True)
        
        sum_indx = qp_data['attention_mask'].sum(dim=-1)
        embedding_indx = 0
        if self.use_model_type == 'gist':
            gist_indx = sum_indx - 2
            embedding_indx = gist_indx
        else:
            eos_indx = sum_indx - 1
            embedding_indx = eos_indx
        for i in range(results.last_hidden_state.shape[0]):
            
            embedding_id = embedding_indx[i].item()
            res_embedding = results.last_hidden_state[i,embedding_id,:]
            res_embedding = res_embedding.unsqueeze(0)
            if self.normlized:
                res_embedding = torch.nn.functional.normalize(res_embedding, dim=-1)
                
            embeddings.append(res_embedding)
        
        all_embeddings = torch.cat(embeddings, dim=0)
       
        return all_embeddings.contiguous()
    
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
    # def forward(self, q_reps: Tensor = None, p_reps: Tensor = None, teacher_score: Tensor = None):
        # print("-----------1-----------------")
        q_reps = self.inference_qp(query)
        # print(f"q_reps:\n{q_reps}")
        # print(f"q_reps shape:{q_reps.shape}")
        p_reps = self.inference_qp(passage)
        # print(f"p_reps:\n{p_reps}")
        # print(f"p_reps size:{p_reps.shape}")
        # exit()

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
