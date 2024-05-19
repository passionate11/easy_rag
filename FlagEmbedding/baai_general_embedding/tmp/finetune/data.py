import math
import os.path
import random
import torch
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer

from .arguments import DataArguments

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []
        # positives negatives
        assert isinstance(self.dataset[item]['positives'], list)
        pos = random.choice(self.dataset[item]['positives'])
        passages.append(pos)

        if len(self.dataset[item]['negatives']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['negatives']))
            negs = random.sample(self.dataset[item]['negatives'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['negatives'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        return query, passages


def get_inputs(input_data, tokenizer, use_model_type, max_length):
    input_ids = [tokenizer.bos_token_id]
    user_tokens=[1786, 4194, 95388]
    
    for message in input_data["messages"]:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)

        if role == "user":
            input_ids += user_tokens + content_ids
            
            if use_model_type=='gist':
                if len(input_ids) >= (max_length - 2):
                    input_ids = input_ids[: max_length-2]
                # 对于user角色，在content后面加上特殊的 <gist> token
                # "<reserved_21>": 110 ->  <gist>
                gist_token_id = tokenizer.encode('<reserved_21>', add_special_tokens=False)
                input_ids += gist_token_id
            else:
                if len(input_ids) >= (max_length - 1):
                    input_ids = input_ids[: max_length-1] 
                      
            input_ids.append(tokenizer.eos_token_id)
            
            attention_mask = [1] * len(input_ids)
            # pad to max len
            input_ids += [tokenizer.eos_token_id] * (
                max_length - len(input_ids)
            )
            attention_mask += [0] * (max_length - len(attention_mask))
    

    return {
            "input_ids": input_ids,
            "attention_mask":attention_mask
        }

def tokenize_qp(tokenizer, qp_data, use_model_type, max_len):
    
    input_ids_list = []
    attention_mask_list = []
    
    for data in qp_data:
        in_text = {
            'messages': [{'role':'user', 'content':'Echo the provided text. Text:{}'.format(data)}]
        }
        
        data_tokenized = get_inputs(in_text, tokenizer, use_model_type, max_len) 
        input_ids_list.append(data_tokenized['input_ids'])
        attention_mask_list.append(data_tokenized['attention_mask'])
        
    return {
            "input_ids": torch.LongTensor([input_ids_list]).view(len(qp_data),-1),
            "attention_mask": torch.LongTensor([attention_mask_list]).view(len(qp_data),-1)
        }


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    use_model_type: str = None

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        # print(f"query:\n{query}")
        # print(f"passage:\n{passage}")
        # exit()
        # TODO
        q_collated = tokenize_qp(self.tokenizer, query, self.use_model_type, self.query_max_len)
        d_collated = tokenize_qp(self.tokenizer,  passage, self.use_model_type,self.passage_max_len)
        
        
        # q_collated = self.tokenizer(
        #     query,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.query_max_len,
        #     return_tensors="pt",
        # )
        # d_collated = self.tokenizer(
        #     passage,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.passage_max_len,
        #     return_tensors="pt",
        # )
        
        # print(f"query:\n{q_collated}")
        # print(f"query input_ids size:{q_collated['input_ids'].shape}")
        # print(f"passage:\n{d_collated}")
        # print(f"passage input_ids size；{d_collated['input_ids'].shape}")
        
        
        return {"query": q_collated, "passage": d_collated}
        # return query_embeddings, passage_embeddings
