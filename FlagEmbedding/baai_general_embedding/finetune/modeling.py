import imp
import logging
from dataclasses import dataclass
from os import truncate
from time import sleep
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
import copy

from my_utils import mix_models
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
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

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
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

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

# 取消batch内pos作为neg
class my_modified_BiEncoderModel(BiEncoderModel):
    TRANSFORMER_CLS = AutoModel

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                # scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                # scores = scores.view(q_reps.size(0), -1)

                # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                # target = target * group_size
                # loss = self.compute_loss(scores, target)
                pos_p_reps_index = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long) \
                    * (group_size)
                neg_mask = torch.ones(p_reps.size(0), device=q_reps.device, dtype=torch.bool)
                neg_mask[pos_p_reps_index] = False
                pos_p_reps = p_reps[pos_p_reps_index]
                neg_p_reps = p_reps[neg_mask]

                # [n*1]
                pos_scores = torch.sum(q_reps * pos_p_reps, dim=-1, keepdim=True)
                # [n * 7]
                neg_scores = self.compute_similarity(q_reps, neg_p_reps)

                scores = torch.cat([pos_scores, neg_scores], dim=-1)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1)

                target = torch.zeros(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                
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

# softmax loss
class my_modified_loss_1_BiEncoderModel(BiEncoderModel):
    TRANSFORMER_CLS = AutoModel

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)
        my_softmax = nn.Softmax(dim=-1)
        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                # scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                # scores = scores.view(q_reps.size(0), -1)

                # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                # target = target * group_size
                # loss = self.compute_loss(scores, target)
                pos_p_reps_index = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long) \
                    * (group_size)
                neg_mask = torch.ones(p_reps.size(0), device=q_reps.device, dtype=torch.bool)
                neg_mask[pos_p_reps_index] = False
                pos_p_reps = p_reps[pos_p_reps_index]
                neg_p_reps = p_reps[neg_mask]

                # [n*1]
                pos_scores = torch.sum(q_reps * pos_p_reps, dim=-1, keepdim=True)
                # [n * 7]
                neg_scores = self.compute_similarity(q_reps, neg_p_reps)

                scores = torch.cat([pos_scores, neg_scores], dim=-1)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1)
                
                scores = my_softmax(scores)
                target = torch.zeros(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)

                scores = my_softmax(scores)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            scores = my_softmax(scores)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )


# balanced batch policy
class my_modified_loss_2_BiEncoderModel(BiEncoderModel):
    TRANSFORMER_CLS = AutoModel

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                # target = target * group_size
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

# 混合层for cocktail_biencoder
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


class Cocktail_BiEncoderModel(BiEncoderModel):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 cocktail_model_list: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__(model_name,
                        normlized,
                        sentence_pooling_method,
                        negatives_cross_device,
                        temperature,
                        use_inbatch_neg,
                )
        
        self.model_list = nn.ModuleList()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        for model in cocktail_model_list:
            cur_model = AutoModel.from_pretrained(model, trust_remote_code=True)
            cur_model = cur_model.to(device)
            self.model_list.append(cur_model)

        # mix model初始化
        self.mix_model = copy.deepcopy(self.model_list[0])
       
        self.weight_list = nn.Parameter(torch.ones(len(self.model_list)))
        

        self.mix_model = self.replace_parameter(self.mix_model, self.model_list, self.weight_list)
        self.mix_model.to(device)
        

    def replace_parameter(self, mix_layer, layer_list, weight_list):
        # weight_list = self.my_softmax(weight_list)
        if isinstance(mix_layer, nn.Linear):
            return MixLayer(layer_list, weight_list)
            
        mix_layer_module = mix_layer._modules
        layer_module_list = [layer._modules for layer in layer_list]
        for name in mix_layer._modules.keys():
            mix_layer_sub_module = mix_layer_module[name]
            layer_sub_module_list = [layer_sub_module[name] for layer_sub_module in layer_module_list]
            setattr(mix_layer, name, self.replace_parameter(mix_layer_sub_module, layer_sub_module_list, weight_list))
        return mix_layer
    
    def encode(self, features):
        if features is None:
            return None
        # print('using mix model encode')
        psg_out = self.mix_model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        print(f'mix weight:{self.weight_list}')

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)
                # print(f"scores.size:{scores.size}")
                # print(f'scores:{scores}')

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                # print(f"target.size:{target.size}")
                # print(f'target:{target}')

                loss = self.compute_loss(scores, target)
                # print('loss back')
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

# 添加head，冻结模型主体参数
class Bi_lmhead_EncoderModel(BiEncoderModel):
    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 freeze_base_model: bool = False,
                 ):
        super().__init__(model_name, normlized, sentence_pooling_method, negatives_cross_device, temperature, use_inbatch_neg)

        hidden_size, laryer_norm_eps, hidden_dropout_prob = self.model.config.hidden_size, self.model.config.layer_norm_eps, self.model.config.hidden_dropout_prob

        self.lm_head = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.LayerNorm(4*hidden_size, eps=laryer_norm_eps),
            nn.Dropout(hidden_dropout_prob),
            nn.GELU(),
            nn.Linear(4*hidden_size, hidden_size)
        )

        if freeze_base_model:
            print("freezing base model parameters")
            for param in self.model.parameters():
                param.requires_grad = False

    def encode(self, features):
        super_encode_func_res = super().encode(features)
        encode_res = self.lm_head(super_encode_func_res)

        if self.normlized:
            encode_res = torch.nn.functional.normalize(encode_res, dim=-1)
        return encode_res

    def encode_sentences(self, sentences, tokenizer, device, max_length=512):
        batch_data = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length,
        ).to(device)
        
        output = self.encode(batch_data)

        return output

class Bi_llm_EncoderModel(BiEncoderModel):
    def __init__(self,
                 model_name: str = None,
                 llm_embedding_token_type: str = 'eos',
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        self.model = model_name
        self.llm_embedding_token_type = llm_embedding_token_type
        
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.negatives_cross_device = negatives_cross_device
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch

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

    def encode(self, features):
        pass