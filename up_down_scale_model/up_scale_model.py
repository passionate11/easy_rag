from lib2to3.pgen2.tokenize import tokenize
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch


input_model_path = '/mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/1_models/Qwen/Qwen1.5-7B-Chat'
targer_layers = 40 
assert targer_layers % 2 == 0, 'targer_layers must be even'
save_path = '/mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/1_models/merged_Qwen1.5_7B-Chat'


model1 = AutoModel.from_pretrained(input_model_path)
model2 = AutoModel.from_pretrained(input_model_path)
tokenizer1 = AutoTokenizer.from_pretrained(input_model_path)

in_num_layers = model1.config.num_hidden_layers
print(f'in_num_layers :{in_num_layers}')
half_targer_layers = targer_layers // 2
model2_start_layer_idx = in_num_layers - half_targer_layers
print(model1)


total_params = sum(p.numel() for p in model1.parameters())
print(f'Total parameters: {total_params}')

merged_model_config = AutoConfig.from_pretrained(input_model_path, num_hidden_layers=targer_layers)
merged_model = AutoModel.from_config(merged_model_config)

for i in range(half_targer_layers):
    merged_model.layers[i].load_state_dict(model1.layers[i].state_dict())

# 加载模型2的后20层权重到新模型的后20层中
for i in range(half_targer_layers):
    merged_model.layers[half_targer_layers + i].load_state_dict(model2.layers[model2_start_layer_idx+i].state_dict())

# 其他属性设置
merged_model.padding_idx = model1.padding_idx
merged_model.vocab_size = model1.vocab_size
merged_model._attn_implementation = model1._attn_implementation
merged_model.norm = model1.norm
merged_model.gradient_checkpointing = model1.gradient_checkpointing


merged_model.save_pretrained(save_path)
tokenizer1.save_pretrained(save_path)

print(merged_model)
total_params = sum(p.numel() for p in merged_model.parameters())
print(f'Total parameters: {total_params}')

