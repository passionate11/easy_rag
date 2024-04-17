<h1 align="center">FlagEmbedding二次开发</h1>
<p align="center">
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made with-Python-purple">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://huggingface.co/C-MTEB">
        <img alt="License" src="https://img.shields.io/badge/C_MTEB-🤗-yellow">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding">
        <img alt="License" src="https://img.shields.io/badge/universal embedding-1.1-red">
    </a>
</p>

<h4 align="center">
    <p>
        <a href=#更新>更新</a> |
        <a href="#项目">项目</a> |
        <a href="#模型列表">模型列表</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>



## 简介
ecom_Rag基于Flagembedding上进行二次开发，致力于提供治理测垂域召回/检索服务，同时包含一些训练/推理trick可供选择


## 更新
- 4/17/2014: 将scripts更名为rag_scripts, 添加2_reranker.py, 支持对多路召回结果进行rerank
    - 支持使用m3-v2-rerank进行精排
    - 输入数据格式{'text': 待检索数据/ ,'metadata': 数据相关信息, 'topk_sim': topk相似分数, 'topk_seed': topk结果, 'topk_metadata': topk元数据}
    - 会根据text和topk_seed中的数据新的相关性对['topk_sim', 'topk_seed', 'topk_metadata']进行重排序
- 4/10/2024: 添加scripts/1_scorer.py，提供方便的检索服务
    - 支持bge/m3两个模型的检索，输入query和documents，输出documents的相似度分数及相关信息 |
    - 输入数据格式:{'text': 待检索数据/doc数据, 'metadata':数据相关信息} | 
    - 输出数据格式:{'text': 待检索数据/ ,'metadata': 数据相关信息, 'topk_sim': topk相似分数, 'topk_seed': topk结果, 'topk_metadata': topk元数据}
- 4/10/2024: 支持bge模型多种trick训练，包括：
    - circle loss
    - Balanced dataset
    - model merging weights training


## 即将支持
- 添加reranker链路代码
- 开发多路召回