from rank_bm25 import BM25Okapi
import jieba
import torch


# 定义文档
docs = [
    "天气不错，我们去公园散步吧。",
    "今天天气真好，适合外出走走。",
    "明天有雨，最好待在家里。"
]

# 定义查询
queries = [
    "去公园散步",
    "天气如何",
    "明天的天气预报"
]

# 分词处理文档
tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
# 分词处理查询
tokenized_queries = [list(jieba.cut(query)) for query in queries]


# 初始化BM25模型
bm25 = BM25Okapi(tokenized_docs)

res = []
# 批量计算每个查询的分数
for query in tokenized_queries:
    scores = bm25.get_scores(query)
    res.append(scores)

res = torch.tensor(res)
print(res)
print(res.shape)
