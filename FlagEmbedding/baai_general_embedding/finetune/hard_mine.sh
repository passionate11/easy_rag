python3 /mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/code_base/FlagEmbedding/FlagEmbedding/baai_general_embedding/finetune/hn_mine.py  \
--model_name_or_path /mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/experiments/3.20_my_bge_large_d5/checkpoint-1557 \
--input_file /mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/data/origin_data/data_for_embedding/3.15_d5d4_new_data/train/bge_train_data/3.19_all_train_data.jsonl \
--output_file /mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/data/origin_data/data_for_embedding/3.15_d5d4_new_data/train/bge_train_data/3.19_all_train_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 