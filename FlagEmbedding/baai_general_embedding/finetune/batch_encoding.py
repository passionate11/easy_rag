from modeling import Bi_llm_head_EncoderModel
from safetensors import safe_open
from transformers import AutoTokenizer
import jsonlines
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

def load_model_and_tokenizer(model_path):
    model = Bi_llm_head_EncoderModel(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    safe_tensor_path = model_path + '/model.safetensors'
    
    model_tensors = {}
    with safe_open(safe_tensor_path, framework='pt', device='cpu') as f:
        for k in f.keys():
            model_tensors[k] = f.get_tensor(k)
    model.load_state_dict(model_tensors)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism")
        model = torch.nn.DataParallel(model)
    return model, tokenizer

def encode(model, tokenizer, features, llm_embedding_token_type, normalized):
    features = {key: value.to(device) for key, value in features.items()}
    
    if isinstance(model, torch.nn.DataParallel):
        psg_out = model.module.model(**features, return_dict=True)
        psg_out = model.module.lm_head(psg_out.last_hidden_state)
    else:
        psg_out = model.model(**features, return_dict=True)
        psg_out = model.lm_head(psg_out.last_hidden_state)
    
    total_len = features['attention_mask'].sum(dim=-1)  # batch size
    if llm_embedding_token_type == 'eos':
        emb_token_idx = total_len - 1
    elif llm_embedding_token_type == 'special':
        emb_token_idx = total_len - 2 
    else:
        raise ValueError('only support [eos, special], contact with hpc for more tech supports')
    
    bsz = psg_out.size(0)
    batch_indices = torch.arange(bsz).to(device)
    p_reps = psg_out[batch_indices, emb_token_idx]
    if normalized:
        p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
    return p_reps.contiguous()

def encode_sentences(model, sentences, tokenizer, max_length=64):
    prompt = '<|im_start|>将下面这个query压缩成一个单词\nquery：{query}\n压缩后的单词：<|emb_0|><|im_end|>'
    prompt_list = [prompt.format(query=data) for data in sentences]
    qp_collated = tokenizer(prompt_list, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encode(model, tokenizer, qp_collated, 'special', True)

def encode_file(in_path, out_path, batch_size, max_length):
    model, tokenizer = load_model_and_tokenizer(model_path)
    with jsonlines.open(in_path, 'r') as reader, jsonlines.open(out_path, 'w') as writer:
        batch_queries = []
        all_data = list(reader)
        progress_bar = tqdm(total=len(all_data))
        for data in all_data:
            batch_queries.append(data['query'])
            if len(batch_queries) >= batch_size:
                res_embeddings = encode_sentences(model, batch_queries, tokenizer, max_length).cpu().tolist()
                for i in range(len(batch_queries)):
                    writer.write({'query': batch_queries[i], 'embedding': res_embeddings[i]})
                batch_queries = []
                progress_bar.update(batch_size)
        if batch_queries:
            res_embeddings = encode_sentences(model, batch_queries, tokenizer, max_length).cpu().tolist()
            for i in range(len(batch_queries)):
                writer.write({'query': batch_queries[i], 'embedding': res_embeddings[i]})
            progress_bar.update(len(batch_queries))

model_path = '/home/admin/workspace/aop_lab/query_rec/experiments/llm_embedding/ckpt_tmp/checkpoint-9'
in_file = '/home/admin/workspace/aop_lab/q_llm_embedding/datas/test_data/hpc_diwen_q2q_test_test_corpus.jsonl'
out_file = in_file.split('.jsonl')[0] + '_embeddingRes.jsonl'
bsz = 64
max_length = 64
encode_file(in_file, out_file, bsz, max_length)