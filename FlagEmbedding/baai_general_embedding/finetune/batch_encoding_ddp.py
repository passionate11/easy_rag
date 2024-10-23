import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from modeling import Bi_llm_head_EncoderModel
from safetensors import safe_open
from transformers import AutoTokenizer
import jsonlines
from tqdm import tqdm
import argparse
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_model_and_tokenizer(model_path, rank):
    device = torch.device(f'cuda:{rank}')
    model = Bi_llm_head_EncoderModel(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    safe_tensor_path = model_path + '/model.safetensors'
    model_tensors = {}
    with safe_open(safe_tensor_path, framework='pt', device='cpu') as f:
        for k in f.keys():
            model_tensors[k] = f.get_tensor(k).to(device)
    model.load_state_dict(model_tensors)
    
    model = DDP(model, device_ids=[rank])
    return model, tokenizer

def encode(model, tokenizer, features, llm_embedding_token_type, normalized, device):
    features = {key: value.to(device) for key, value in features.items()}
    
    if isinstance(model, DDP):
        psg_out = model.module.model(**features, return_dict=True)
        psg_out = model.module.lm_head(psg_out.last_hidden_state)
    else:
        psg_out = model.model(**features, return_dict=True)
        psg_out = model.lm_head(psg_out.last_hidden_state)
    
    total_len = features['attention_mask'].sum(dim=-1)
    emb_token_idx = total_len - 1 if llm_embedding_token_type == 'eos' else total_len - 2
    
    bsz = psg_out.size(0)
    batch_indices = torch.arange(bsz).to(device)
    p_reps = psg_out[batch_indices, emb_token_idx]
    if normalized:
        p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
    return p_reps.contiguous()

def encode_sentences(model, sentences, tokenizer, device, max_length=64):
    prompt = '<|im_start|>将下面这个query压缩成一个单词\nquery：{query}\n压缩后的单词：<|emb_0|><|im_end|>'
    prompt_list = [prompt.format(query=data) for data in sentences]
    qp_collated = tokenizer(prompt_list, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encode(model, tokenizer, qp_collated, 'special', True, device)

def main(rank, world_size, model_path, in_path, out_path, batch_size, max_length):
    setup(rank, world_size)
    model, tokenizer = load_model_and_tokenizer(model_path, rank)
    device = torch.device(f'cuda:{rank}')

    # 每个进程都读取数据
    with jsonlines.open(in_path, 'r') as reader:
        all_data = list(reader)

    # 每个进程计算处理的数据片段
    num_samples_per_rank = len(all_data) // world_size
    start_idx = rank * num_samples_per_rank
    end_idx = start_idx + num_samples_per_rank if rank != world_size - 1 else len(all_data)
    local_data = all_data[start_idx:end_idx]

    if rank == 0:
        writer = jsonlines.open(out_path, 'w')
    
    batch_queries = []
    progress_bar = tqdm(total=len(local_data), desc=f"Rank {rank}") if rank == 0 else None
    for data in local_data:
        batch_queries.append(data['query'])
        if len(batch_queries) >= batch_size:
            res_embeddings = encode_sentences(model, batch_queries, tokenizer, device, max_length).cpu().tolist()
            if rank == 0:
                for i in range(len(batch_queries)):
                    writer.write({'query': batch_queries[i], 'embedding': res_embeddings[i]})
            batch_queries = []
            if rank == 0:
                progress_bar.update(batch_size)
    if batch_queries:
        res_embeddings = encode_sentences(model, batch_queries, tokenizer, device, max_length).cpu().tolist()
        if rank == 0:
            for i in range(len(batch_queries)):
                writer.write({'query': batch_queries[i], 'embedding': res_embeddings[i]})
        if rank == 0:
            progress_bar.update(len(batch_queries))

    if rank == 0:
        writer.close()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc_per_node', type=int, default=8, help='Number of processes per node')
    args = parser.parse_args()
    
    model_path = '/home/admin/workspace/aop_lab/query_rec/experiments/llm_embedding/ckpt_tmp/checkpoint-9'
    in_file = '/home/admin/workspace/aop_lab/q_llm_embedding/datas/test_data/hpc_diwen_q2q_test_test_corpus.jsonl'
    out_file = in_file.split('.jsonl')[0] + '_embeddingRes.jsonl'
    bsz = 64
    max_length = 64

    world_size = args.nproc_per_node
    mp.spawn(main, args=(world_size, model_path, in_file, out_file, bsz, max_length), nprocs=world_size, join=True)