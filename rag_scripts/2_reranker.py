import jsonlines
import time
from torch import cuda
import torch
from tqdm import tqdm
from collections import defaultdict
import os
import argparse
from FlagEmbedding import BGEM3FlagModel, FlagReranker, LayerWiseFlagLLMReranker
from sentence_transformers import SentenceTransformer


device = 'cuda' if cuda.is_available() else 'cpu'

start_time = time.time()
# 承接1_reranker 
# 输入数据格式{'text': '检索数据', 'metadata': '元数据', topk_sim: '检索结果的相似度分数', 'topkseed': '检索结果', 'topk_metadata': '检索结果的元数据'}
# topseed是必须的,需要根据text和topseed重排 topseed和topmetadata
class Reranker():
    def __init__(self, model_path, model_type, batch_size, topk, in_to_rerank_data_path, out_reranked_path):
        self.model_path = model_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.topk = topk
        self.in_to_rerank_data_path = in_to_rerank_data_path
        self.out_reranked_path = out_reranked_path
        
        self.in_to_rerank_data_path_list = in_to_rerank_data_path.split(',')
        if len(self.in_to_rerank_data_path_list) > 1:
            self.multi_recall = True
        else:
            self.multi_recall = False
    
    
    # utils
    def read_data(self, data_path):
        data = []
        with jsonlines.open(data_path, 'r') as reader:
            for tmp in reader:
                data.append(tmp)
        return data
    # utils
    def rank_list_by_index(self, in_list, index_list):
        return [in_list[i] for i in index_list]

    # 通过m3计算rerank分数
    def m3_reranker_v2_score(self, rerank_model, normalize, batch_data):
        batch_size = len(batch_data)
        scores = rerank_model.compute_score(batch_data, normalize=normalize, batch_size=batch_size, max_length=1024)
        return scores

    def minicpm_reranker_score(self, rerank_model, normalize, batch_data):
        batch_size = len(batch_data)
        scores = rerank_model.compute_score(batch_data, normalize=normalize, batch_size=batch_size, max_length=1024)
        return scores

    # bge/m3等
    def biencoder_reranker_scorer(self, rerank_model, rerank_score_func):
        if self.multi_recall is True:    # 如果是多路召回
            all_data = []

            for idx, data_path in enumerate(self.in_to_rerank_data_path_list):
                
                if idx == 0:
                    all_data = self.read_data(data_path)
                else:
                    cur_all_data = self.read_data(data_path)
                    for tmp_idx, tmp_data in enumerate(cur_all_data):
                        all_data[tmp_idx]['topk_sim'].extend(tmp_data['topk_sim'])
                        all_data[tmp_idx]['topk_seed'].extend(tmp_data['topk_seed'])
                        all_data[tmp_idx]['topk_metadata'].extend(tmp_data['topk_metadata'])
        else:
            all_data = self.read_data(self.in_to_rerank_data_path)

        all_top_seed = []
        for tmp in all_data:
            all_top_seed.append(tmp['topk_seed'])


        with jsonlines.open(self.out_reranked_path, 'w') as writer:
            all_len = len(all_data)
            progress_bar = tqdm(total=all_len, desc="reranking...")

            # 由于每条数据有n个待重排句子，所以minibatch为 batch_size / n 
            mini_batch = 1 if self.batch_size < len(all_top_seed[0]) else self.batch_size // len(all_top_seed[0])
            print(f'your minibatch is {mini_batch}')
            for i in range(0, all_len, mini_batch):
                batch_data = all_data[i: i+mini_batch]
                mini_batch = len(batch_data)    # 最后如果不够batch会越界，所以这里mini_batch设为batch_data的长度，保证不越界
                # 包含 minibatch * n 条数据
                batch_to_cal_data = []
                for j in range(mini_batch):
                    for ttmp in range(len(all_top_seed[0])):
                        batch_to_cal_data.append([batch_data[j]['text'], batch_data[j]['topk_seed'][ttmp]])

                cur_scores = rerank_score_func(rerank_model, True, batch_to_cal_data)
                scores = torch.tensor(cur_scores)
                real_scores = []
                real_indices = []
                for ttmp in range(mini_batch):
                    tmp_score = scores[ttmp*len(all_top_seed[0]) :(ttmp+1)*len(all_top_seed[0])]
                    top_k_sim, top_k_indices = torch.topk(tmp_score, k=self.topk)
                    real_scores.append(top_k_sim.tolist())
                    real_indices.append(top_k_indices.tolist())

        
                for j in range(mini_batch):
                    cur_scores, cur_indices = real_scores[j], real_indices[j]
                    to_rank_key_list = ['topk_sim', 'topk_seed', 'topk_metadata']

                    for to_rank_key in to_rank_key_list:
                        batch_data[j][to_rank_key] = self.rank_list_by_index(batch_data[j][to_rank_key], cur_indices)
                    batch_data[j]['rerank_score'] = cur_scores
                    writer.write(batch_data[j])

                progress_bar.update(mini_batch)

    # 调用函数
    def get_rerank_score(self):
        print(f'********** Using {self.model_type} for rerank **********')
        if self.model_type == 'm3-reranker-v2':
            rerank_model = FlagReranker(self.model_path, use_fp16=True)
            self.biencoder_reranker_scorer(rerank_model, self.m3_reranker_v2_score)
        elif self.model_type == 'minicpm-rearnker':
            rerank_model = LayerWiseFlagLLMReranker(self.model_path, use_bf16=True)
            self.biencoder_reranker_scorer(rerank_model, self.minicpm_reranker_score)

parser = argparse.ArgumentParser(description='Reranker启动！')

parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
parser.add_argument('--model_type', type=str, required=True, help='模型种类')
parser.add_argument('--batch_size', type=int, default=128, help='批处理大小，默认为32')
parser.add_argument('--topk', type=int, default=20, help='仅考虑排名前k的结果，默认为20')
parser.add_argument('--in_to_rerank_data_path', type=str, required=True, help='输入待rerank文件的路径')
parser.add_argument('--out_reranked_path', type=str, required=True, help='输出rerank结果路径')

args = parser.parse_args()

model_path = args.model_path
model_type = args.model_type
batch_size = args.batch_size
topk = args.topk
in_to_rerank_data_path = args.in_to_rerank_data_path
out_reranked_path = args.out_reranked_path

my_ranker = Reranker(model_path, model_type, batch_size, topk, in_to_rerank_data_path, out_reranked_path)
my_ranker.get_rerank_score()

end_time = time.time()
duration_time = end_time - start_time
duration_time = duration_time / 60
print(f'程序执行时间: {duration_time} min')
