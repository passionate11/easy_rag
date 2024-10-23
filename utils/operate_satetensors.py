import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import safetensors

import torch

# 改名
def rename_safetensors_key(in_path, out_path):
    
    with safe_open(in_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    
    original_data = load_file(in_path)
    # 创建一个新的字典，用于存储修改后的键值对
    updated_data = {}
    # 遍历原始字典，对每个键进行修改，并将修改后的键值对存储到新的字典中
    for key in original_data.keys():
        if 'model.' in key:
            new_key = 'model.' + key.split('model.')[-1]  # 修改键名
        else:
            new_key = key
        updated_data[new_key] = original_data[key]
    # 将新的字典以及原始的metadata数据保存到新的safetensors文件中
    save_file(tensors=updated_data, filename=out_path, metadata=metadata)



in_path = '/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/reranker/list_wise/qwen-0.5/4.2_50w_base/checkpoints/checkpoint-1500/model.safetensors'
out_path = '/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/reranker/list_wise/qwen-0.5/4.2_50w_base/checkpoints/checkpoint-1500/model_modified.safetensors'
rename_safetensors_key(in_path, out_path)
# show key
def show_safetensors_key(in_path):
    with safe_open(in_path, framework="pt", device="cpu") as f:
        print(f.metadata())
        # print(f.keys())
        print(type(f))

show_safetensors_key(out_path)


# model_param = safetensors.load_safetensors(in_path)
# for k, v in model_param.items():
#     new_k = k[6:]
#     model_param[k] = model_param.pop(k)