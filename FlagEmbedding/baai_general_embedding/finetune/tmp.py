from modeling import Bi_llm_head_EncoderModel
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

# 假设自定义模型 Bi_llm_head_EncoderModel 已经定义好
model_path = '/home/admin/workspace/aop_lab/query_rec/experiments/llm_embedding/ckpt_tmp/checkpoint-8'

# 1. 首先加载模型结构
model = Bi_llm_head_EncoderModel(model_path)

safe_tensor_path = model_path + '/model.safetensors'
model_tensors = {}
with safe_open(safe_tensor_path, framework='pt',device='cpu') as f:
    for k in f.keys():
        model_tensors[k] = f.get_tensor(k)
model.load_state_dict(model_tensors)

print(model)