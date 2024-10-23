from transformers import AutoTokenizer

# 加载预训练的 tokenizer，例如 BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 打印原始词汇表大小
print("Original vocab size:", len(tokenizer))

# 定义要添加的特殊字符
special_tokens = {'additional_special_tokens': ['<|emb_0|>']}

# 添加特殊字符
tokenizer.add_special_tokens(special_tokens)

# 打印更新后的词汇表大小
print("Updated vocab size:", len(tokenizer))

# 保存更新后的 tokenizer
tokenizer.save_pretrained("path/to/save/your/tokenizer")

# 使用更新后的 tokenizer
encoded_input = tokenizer("This is a test sentence with [MY_SPECIAL_TOKEN].")
print(encoded_input)
