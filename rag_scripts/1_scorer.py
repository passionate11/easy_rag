
from statistics import mode
import jsonlines
import time
from torch import cuda
import torch
from tqdm import tqdm
from collections import defaultdict
import os
import argparse
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel

device = 'cuda' if cuda.is_available() else 'cpu'

start_time = time.time()
class Scorer():
    '''
    model_path: 模型路径
    model_type: 模型类型
    batch_size: batch_size 
    save_intermediate_res: true/false 是否保存中间结果：包括test_data 的embedding 和 seed_data 的embedding
    topk: 检索topk
    test_data: 待检索数据
    seed_data: 检索库数据
    out_score_path: 输出结果路径
    # 输入数据格式{'text': '待检索数据', 'metadata': {所有的数据相关信息}} 用test中的text检索seed中的text
    # ！！ 注意，如果数据是chunk的数据,比如前20条数据是同一个数据的chunk,最终的topk为这一条的数据的topk,那么需要在metadata中添加new_id字段
    # 输出数据格式:{'text': '检索数据', 'metadata': '元数据', topk_sim: '检索结果的相似度分数', 'topkseed': '检索结果', 'topk_metadata': '检索结果的元数据'}
    '''
    def __init__(self, model_path, model_type, batch_size, topk, save_intermediate_res, in_test_data, in_seed_data, out_score_path):  
        self.model_path = model_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.topk = topk
        self.save_intermediate_res = save_intermediate_res
        self.in_test_data = in_test_data
        self.in_seed_data = in_seed_data
        self.out_score_path = out_score_path
        if self.save_intermediate_res:
            present_dir = os.path.dirname(self.out_score_path)
            self.inter_test_embedding_path = present_dir + '/inter_test_embedding.jsonl'
            self.inter_seed_embedding_path = present_dir + '/inter_seed_embedding.jsonl'

    # utils
    def data_reader(self, data_path):
        data = []
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                data.append(obj)
        return data

    # utils
    def compute_cosine_similarity(self, vec1, vec2):
        return (vec1@vec2.T)
    
    # from yy
    def data_pp(self, instruction, tokenizer, eos_token, samples, max_len=8192):
        instruction = instruction + eos_token
        instruction_left, instruction_right = instruction.split("{}")
        token_left = tokenizer.tokenize(instruction_left)
        token_right = tokenizer.tokenize(instruction_right)
        len_instruction = len(token_left) + len(token_right)
        batch_ids, batch_attn = [], []
        cur_max_len = 0
        for sample in samples:
            token = tokenizer.tokenize(sample)[:max_len - len_instruction]
            ids_all = tokenizer.convert_tokens_to_ids(token_left + token + token_right)
            attn = [1] * len(ids_all) + [0] * (max_len - len(ids_all))
            cur_max_len = max(cur_max_len, len(ids_all))
            ids_all += [tokenizer.pad_token_id] * (max_len - len(ids_all))
            batch_ids.append(ids_all)
            batch_attn.append(attn)
        batch_ids = torch.tensor([x[:cur_max_len] for x in batch_ids])
        batch_attn = torch.tensor([x[:cur_max_len] for x in batch_attn])
        inputs = {"input_ids": batch_ids, "attention_mask": batch_attn}
        emb_token_idx = batch_attn.sum(-1) - 1
        return inputs, emb_token_idx

    # encode方法记得不要返回张量哦，转为list，.cpu().tolist()
    # encode_method
    def bge_encode(self, model, sentences):
        return model.encode(sentences)
        
    def m3_dense_encode(self, model, sentences):
        return model.encode(sentences, return_dense=True)['dense_vecs']
    
    def qwen_instruction_encode(self, model, tokenizer, sentences):
        instruction_dict = {"emb_instruct_prefix": "请将下面的文本\"", "emb_instruct_postfix": "\"转换成一个词："}
        instruction = instruction_dict["emb_instruct_prefix"] + "{}" + instruction_dict["emb_instruct_postfix"]
        eos_token = tokenizer.eos_token
        encoded_input, emb_token_idx = self.data_pp(instruction, tokenizer, eos_token, sentences)
        for k, v in encoded_input.items():
            encoded_input[k] = v.to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)[0]
            embeddings = model_output[torch.arange(model_output.size(0)).unsqueeze(1), emb_token_idx.unsqueeze(1), :].squeeze()
        
        embeddings = embeddings.cpu().tolist()
        return embeddings

    def m3_sparse_encode(self, model, sentences):
        return model.encode(sentences, return_dense=False, return_sparse=True, return_colbert_vecs=False)

    def m3_multi_vector_encode(self, model, sentences):
        model.encode(sentences, return_dense=False, return_sparse=False, return_colbert_vecs=True)


    # m3是混合检索，直接得出最后的结果，TODO 考虑添加dense模式
    def m3_score(self, model, sentence_pairs, weights_for_different_modes=[0.4, 0.2, 0.4]):
        return model.compute_score(sentence_pairs, 
                        max_passage_length=128, # a smaller max length leads to a lower latency
                        weights_for_different_modes=weights_for_different_modes)['dense']
                        #   ['colbert+sparse+dense']

    # 会根据不同的模型类别调用不同的scorer，目前支持bge检索和m3的混合检索


    # 将数据embedding化
    def embedding_data(self, model, encode_func, mode, tokenizer=None):
        if mode == 'test':
            cur_data = self.data_reader(self.in_test_data)
            inter_save_path = self.inter_test_embedding_path
        elif mode == 'seed':
            cur_data = self.data_reader(self.in_seed_data)
            inter_save_path = self.inter_seed_embedding_path
        else:
            print('wrong embedding data mode!')
            exit()
        total_data_lens = len(cur_data)
        progressbar_embedding = tqdm(total=len(cur_data), desc=f"embedding {mode} data...")

        all_text = []
        all_metadata = []
        all_embedding = []
        for i in range(0, total_data_lens, self.batch_size):
            batch_sent, batch_metadata, batch_embedding = [], [], []

            for j in range(self.batch_size):
                if i + j > total_data_lens - 1:
                    break 
                batch_sent.append(cur_data[i+j]['text'])
                batch_metadata.append(cur_data[i+j]['metadata'])
            
            if hasattr(model, 'config') and 'qwen' in model.config.model_type:
                # print('tokenizer is used, make sure is right behavior~')
                batch_embedding = encode_func(model, tokenizer, batch_sent)
            else:
                batch_embedding = encode_func(model, batch_sent)

            # 保存中间结果
            if self.save_intermediate_res is True:
                with jsonlines.open(inter_save_path, 'a') as writer:
                    print(f'save to inter {mode}...')
                    for j in range(len(batch_sent)):
                        writer.write({'text': batch_sent[j], 'metadata': batch_metadata[j], 'embedding': batch_embedding[j].tolist(), })
                
            all_text.extend(batch_sent)
            all_metadata.extend(batch_metadata)
            all_embedding.extend(batch_embedding)
            # 更新进度条
            if i + self.batch_size > total_data_lens:       
                progressbar_embedding.update(total_data_lens-i)
            else:
                progressbar_embedding.update(self.batch_size)

        return all_text, all_metadata, all_embedding
    
    # 通过计算embedding 获取 score
    def embedding_scorer(self, model, encode_func, tokenizer=None):

        test_text, test_metadata, test_embedding = self.embedding_data(model, encode_func, 'test', tokenizer)
        print(type(test_embedding))
        print(len(test_embedding))
        test_embedding = torch.tensor(test_embedding, dtype=torch.float32, device=device)
        test_embedding_normalized = torch.nn.functional.normalize(test_embedding)
        
        seed_text, seed_metadata, seed_embedding = self.embedding_data(model, encode_func, 'seed', tokenizer)
        seed_embedding = torch.tensor(seed_embedding, dtype=torch.float32, device=device)
        seed_embedding_normalized = torch.nn.functional.normalize(seed_embedding)
        
        similarities = self.compute_cosine_similarity(test_embedding_normalized, seed_embedding_normalized)
        print(f"similarities's shape is :{similarities.shape}" )

        top_k_similarities, top_k_indices = torch.topk(similarities, self.topk, dim=1)
        top_k_similarities = top_k_similarities.tolist()
        top_k_indices = top_k_indices.tolist()

        # 将结果聚合
        with jsonlines.open(self.out_score_path, 'w') as writer:
            results = defaultdict(list)

            new_id_cnt = 0
            for i, cur_test_metadata in enumerate(tqdm(test_metadata, desc="writing results...")):
                # topk seed mess
                topk_metadata, topk_seed = [], []
                for tmp in top_k_indices[i]:
                    topk_metadata.append(seed_metadata[tmp])
                    topk_seed.append(seed_text[tmp])


                zip_data = zip(top_k_similarities[i], topk_seed, topk_metadata, [cur_test_metadata]*len(topk_metadata), [test_text[i]]*len(topk_metadata))

                if test_metadata[0].get('new_id'):
                    results[cur_test_metadata['new_id']].extend(zip_data)
                else:
                    results[new_id_cnt].extend(zip_data)
                    new_id_cnt += 1
        
            # 将结果排序并写入
            for new_id, cur_res in results.items():
                cur_res.sort(key=lambda x: x[0], reverse=True)

                topk_sim, topk_seed, topk_metadata, cur_test_metadata, cur_test_text = zip(*cur_res[:self.topk])
                cur_test_metadata = cur_test_metadata[0]
                cur_test_text = cur_test_text[0]
                to_write_data = {
                    'text': cur_test_text,
                    'metadata': cur_test_metadata,
                    'topk_sim': topk_sim,
                    'topk_seed': topk_seed,
                    'topk_metadata': topk_metadata,
                }
                writer.write(to_write_data)

    # 有的模型直接输出score，没有中间embedding
    def no_embedding_scorer(self, model, score_func):
        # 如果模型直接输出分数而不是embedding，则不能保存中间embedding结果
        self.sava_intermediate_res = False

        # TODO 这里记得补全
        test_data = self.data_reader(self.in_test_data)
        seed_data = self.data_reader(self.in_seed_data)

        

        test_text, test_metadata = [], []
        seed_text, seed_metadata = [], [] 
        for tmp in test_data:
            test_text.append(tmp['text'])
            test_metadata.append(tmp['metadata'])
        for tmp in seed_data:
            seed_text.append(tmp['text'])
            seed_metadata.append(tmp['metadata'])

        score_list = []
        progress_bar = tqdm(total=len(test_data), desc="cal m3 sim scores...")

        # 为了计算方便，这里的marco_batch 为self.batch_size //  len(seed_data)
        marco_batch = 1 if self.batch_size < len(seed_data) else self.batch_size // len(seed_text)
        
        for marco_idx in range(0, len(test_text), marco_batch):
            cur_batch = []
            for tmp in range(marco_batch):
                for cur_seed_data in seed_text:
                    cur_batch.append([test_text[marco_idx+tmp], cur_seed_data])

            
            cur_scores = score_func(model=model, sentence_pairs=cur_batch)            
     
            seed_len = len(cur_scores) / marco_batch
            seed_len = int(seed_len)

            for tmp in range(marco_batch):
                score_list.append(cur_scores[tmp*seed_len:(tmp+1)*seed_len])

            progress_bar.update(marco_batch)


        results = defaultdict(list)
        
        score_list = torch.tensor(score_list)
        
        top_k_similarities, top_k_indices = torch.topk(score_list, self.topk, dim=1)
        top_k_similarities = top_k_similarities.tolist()
        top_k_indices = top_k_indices.tolist()


        # 将结果聚合
        with jsonlines.open(self.out_score_path, 'w') as writer:
            results = defaultdict(list)

            new_id_cnt = 0
            for i, cur_test_metadata in enumerate(tqdm(test_metadata, desc="writing results...")):
                # topk seed mess
                topk_metadata, topk_seed = [], []
                for tmp in top_k_indices[i]:
                    topk_metadata.append(seed_metadata[tmp])
                    topk_seed.append(seed_text[tmp])


                zip_data = zip(top_k_similarities[i], topk_seed, topk_metadata, [cur_test_metadata]*len(topk_metadata), [test_text[i]]*len(topk_metadata))

                if test_metadata[0].get('new_id'):
                    results[cur_test_metadata['new_id']].extend(zip_data)
                else:
                    results[new_id_cnt].extend(zip_data)
                    new_id_cnt += 1
        
            # 将结果排序并写入
            for new_id, cur_res in results.items():
                cur_res.sort(key=lambda x: x[0], reverse=True)

                topk_sim, topk_seed, topk_metadata, cur_test_metadata, cur_test_text = zip(*cur_res[:self.topk])
                cur_test_metadata = cur_test_metadata[0]
                cur_test_text = cur_test_text[0]
                to_write_data = {
                    'text': cur_test_text,
                    'metadata': cur_test_metadata,
                    'topk_sim': topk_sim,
                    'topk_seed': topk_seed,
                    'topk_metadata': topk_metadata,
                }
                writer.write(to_write_data)

    def get_retrieval_score(self):
        if self.model_type == 'bge':
            print(f'********** using bge_kind_model for scorer **********')
            model = SentenceTransformer(self.model_path, device=device)
            self.embedding_scorer(model, self.bge_encode)
        elif self.model_type == 'm3_dense':
            print(f'********** using m3_dense for scorer **********')
            model = BGEM3FlagModel(self.model_path, use_fp16=True)
            self.embedding_scorer(model, self.m3_dense_encode)
        elif self.model_type == 'm3_sparse':    # TODO sparse和multivector计算最终相似度的分数不一样，这里之后再支持吧
            print(f'********** using m3_sparse for scorer **********')
            model = BGEM3FlagModel(self.model_path, use_fp16=True)
            self.embedding_scorer(model, self.m3_sparse_encode)
        elif self.model_type == 'm3_multivector':
            print(f'********** using m3_multivector for scorer **********')
            model = BGEM3FlagModel(self.model_path, use_fp16=True)
            self.embedding_scorer(model, self.m3_multi_vector_encode)
        elif self.model_type == 'm3_score':
            print(f'********** using m3_kind_model for scorer **********')
            model = BGEM3FlagModel(self.model_path, use_multivector=False, use_sparse=False, use_fp16=True)
            self.no_embedding_scorer(model, self.m3_score)
        elif self.model_type == 'qwen_instruction':
            print(f'********** using qwen_instruction for scorer **********')
            config = AutoConfig.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # model = AutoModelForCausalLM.from_pretrained(self.model_path, config=config)
            # model = model.model
            model = AutoModel.from_pretrained(self.model_path, config=config)
            model = model.to(device)
            model.eval()
            self.embedding_scorer(model, self.qwen_instruction_encode, tokenizer)
        else:
            print(f'the model type is wrong ! Please recheck the parameter !')
    


parser = argparse.ArgumentParser(description='启动！')

parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
parser.add_argument('--model_type', type=str, required=True, help='模型种类')
parser.add_argument('--batch_size', type=int, default=128, help='批处理大小，默认为32')
parser.add_argument('--topk', type=int, default=20, help='仅考虑排名前k的结果，默认为20')
parser.add_argument('--save_inter_res', default=True, help='是否保存中间结果，默认为是')
parser.add_argument('--test_data', type=str, required=True, help='测试数据集的路径')
parser.add_argument('--seed_data', type=str, required=True, help='检索数据集的路径')
parser.add_argument('--out_score_path', type=str, required=True, help='输出分数文件的路径')

args = parser.parse_args()

model_path = args.model_path
model_type = args.model_type
batch_size = args.batch_size
topk = args.topk
save_inter_res = args.save_inter_res
test_data = args.test_data
seed_data = args.seed_data
out_score_path = args.out_score_path


scorer = Scorer(
    model_path=model_path, 
    model_type=model_type, 
    batch_size=batch_size,
    topk=topk, 
    save_intermediate_res=save_inter_res, 
    in_test_data=test_data, 
    in_seed_data=seed_data, 
    out_score_path=out_score_path
    )
scorer.get_retrieval_score()

end_time = time.time()
duration_time = end_time - start_time
duration_time = duration_time / 60
print(f'程序执行时间: {duration_time} min')
