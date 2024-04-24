from transformers import AutoModel


model_in = AutoModel.from_pretrained('/mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/1_models/Qwen/Qwen1.5-7B-Chat')

print(model_in)


total_params = sum(p.numel() for p in model_in.parameters())
trainable_params = sum(p.numel() for p in model_in.parameters() if p.requires_grad)
print(f'Total parameters: {total_params}')
print(f'Trainable parameters: {trainable_params}')
