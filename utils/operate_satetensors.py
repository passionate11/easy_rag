import torch
from safetensors import safe_open
from safetensors.torch import save_file


import torch

# 改名
def rename_safetensors_key(in_path, out_path):
    
    tensors = {}
    with safe_open(in_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            new_key = key[6:]
            tensors[new_key] = f.get_tensor(key)
        
    save_file(tensors, out_path)
in_path = '/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/reranker/list_wise/qwen-0.5/4.2_50w_base/checkpoints/checkpoint-1500/model.safetensors'
out_path = '/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/reranker/list_wise/qwen-0.5/4.2_50w_base/checkpoints/checkpoint-1500/model_modified.safetensors'
# rename_safetensors_key(in_path, out_path)
# show key
def show_safetensors_key(in_path):
    with safe_open(in_path, framework="pt", device="cpu") as f:
        print(f.keys())

show_safetensors_key(out_path)