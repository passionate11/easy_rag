from typing import cast, List, Union, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, is_torch_npu_available


class FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.stack(all_embeddings)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d

class LLM_FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            use_model_type: str = None,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = False
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        # , attn_implementation='flash_attention_2'
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.use_model_type = use_model_type

        self.device = torch.device("cuda")
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # elif is_torch_npu_available():
        #     self.device = torch.device("npu")
        # else:
        #     self.device = torch.device("cpu")
        #     use_fp16 = False
        # if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            # TODO 在这里修改输入文本的格式
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def get_inputs(self, input_data, tokenizer, use_model_type, max_length):
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

    def tokenize_qp(self, tokenizer, qp_data, use_model_type, max_len):
    
        input_ids_list = []
        attention_mask_list = []
        
        for data in qp_data:
            in_text = {
                'messages': [{'role':'user', 'content':'Echo the provided text. Text:{}'.format(data)}]
            }
            
            data_tokenized = self.get_inputs(in_text, tokenizer, use_model_type, max_len) 
            input_ids_list.append(data_tokenized['input_ids'])
            attention_mask_list.append(data_tokenized['attention_mask'])
            
        return {
                "input_ids": torch.LongTensor([input_ids_list]).view(len(qp_data),-1).to('cuda'),
                "attention_mask": torch.LongTensor([attention_mask_list]).view(len(qp_data),-1).to('cuda')
            }

    def inference_qp(self, qp_data):
        model = self.model.model
        embeddings = [] # 存储所有数据的embedding
        cur_in_ids = {
                        'input_ids': qp_data['input_ids'],
                        'attention_mask':qp_data['attention_mask']
                        }
        
        results = model(**cur_in_ids,return_dict=True)
        # print(f"result:\n{results}")
        sum_indx = qp_data['attention_mask'].sum(dim=-1)
        # print(f"sum_indx:\n{sum_indx}")
        embedding_indx = 0
        if self.use_model_type == 'gist':
            # print(f"$$$$$$$$$1$$$$$$$$")
            gist_indx = sum_indx - 2
            embedding_indx = gist_indx
        else:
            # print(f"$$$$$$$$$2$$$$$$$$")
            eos_indx = sum_indx - 1
            embedding_indx = eos_indx
        # print(f"embedding_indx:{embedding_indx}")
        for i in range(results.last_hidden_state.shape[0]):
            
            embedding_id = embedding_indx[i].item()
            res_embedding = results.last_hidden_state[i,embedding_id,:]
            res_embedding = res_embedding.unsqueeze(0)
            # if self.normlized:
            #     res_embedding = torch.nn.functional.normalize(res_embedding, dim=-1)
                
            embeddings.append(res_embedding)
        
        all_embeddings = torch.cat(embeddings, dim=0)
       
        return all_embeddings.contiguous()

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            # print("------------1-------------")
            # print(f"sentences_batch shape:\n{len(sentences_batch)}")
            # TODO 更改tokenizer的格式
            # TODO 把use_model_type传进来
            inputs = self.tokenize_qp(self.tokenizer, sentences_batch, self.use_model_type, max_length)
            # print(f"inputs['input_ids'] shape:{inputs['input_ids'].shape}\n")
            # print(f"inputs['attention_mask'] shape:{inputs['attention_mask'].shape}\n")
            # TODO 获取一个batch中所有数据的embedding
            embeddings = self.inference_qp(inputs)
            # print(f"embeddings shape:\n{embeddings.shape}")
            # last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            # embeddings = self.pooling(last_hidden_state, inputs['attention_mask']) #相当于存储了一个batch的句子的embedding
            
            
            if self.normalize_embeddings:
                # print("------------2-------------")
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                # print(f"embeddings(normalize_embeddings) shape:\n{embeddings.shape}")
            embeddings = cast(torch.Tensor, embeddings)
            # print(f"embeddings(normalize_embeddings &cast) shape:\n{embeddings.shape}")

            if convert_to_numpy:
                # print("------------3-------------")
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            # print("------------4-------------")
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            # print(f"embeddings(concatenate) shape:\n{embeddings.shape}")
        else:
            # print("------------5-------------")
            all_embeddings = torch.stack(all_embeddings)
            # print(f"embeddings(stack) shape:\n{embeddings.shape}")

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d


class LLMEmbedder:
    instructions = {
        "qa": {
            "query": "Represent this query for retrieving relevant documents: ",
            "key": "Represent this document for retrieval: ",
        },
        "convsearch": {
            "query": "Encode this query and context for searching relevant passages: ",
            "key": "Encode this passage for retrieval: ",
        },
        "chat": {
            "query": "Embed this dialogue to find useful historical dialogues: ",
            "key": "Embed this historical dialogue for retrieval: ",
        },
        "lrlm": {
            "query": "Embed this text chunk for finding useful historical chunks: ",
            "key": "Embed this historical text chunk for retrieval: ",
        },
        "icl": {
            "query": "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },
        "tool": {
            "query": "Transform this user request for fetching helpful tool descriptions: ",
            "key": "Transform this tool description for retrieval: "
        },
    }

    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            use_fp16: bool = True
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False

        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 256,
                       task: str = 'qa') -> np.ndarray:
        '''
        Encode queries into dense vectors. 
        Automatically add instructions according to given task.
        '''
        instruction = self.instructions[task]["query"]

        if isinstance(queries, str):
            input_texts = instruction + queries
        else:
            input_texts = [instruction + q for q in queries]

        return self._encode(input_texts, batch_size=batch_size, max_length=max_length)

    def encode_keys(self, keys: Union[List[str], str],
                    batch_size: int = 256,
                    max_length: int = 512,
                    task: str = 'qa') -> np.ndarray:
        '''
        Encode keys into dense vectors. 
        Automatically add instructions according to given task.
        '''
        instruction = self.instructions[task]["key"]

        if isinstance(keys, str):
            input_texts = instruction + keys
        else:
            input_texts = [instruction + k for k in keys]
        return self._encode(input_texts, batch_size=batch_size, max_length=max_length)

    @torch.no_grad()
    def _encode(self, sentences: Union[List[str], str], batch_size: int = 256, max_length: int = 512) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} not implemented!")

