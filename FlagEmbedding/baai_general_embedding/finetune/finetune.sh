# 模型路径 模型输出路径 数据路径 epoch数量
model_path=/home/admin/workspace/aop_lab/models/Qwen/Qwen2-7B-Instruct
# model_path=/home/admin/workspace/aop_lab/models/BAAI/bge-large-zh-v1.5
model_output_dir=/home/admin/workspace/aop_lab/query_rec/experiments/llm_embedding/ckpt_tmp
# data_dir=/home/admin/workspace/aop_lab/query_rec/experiments/llm_embedding/toy_finetune_data.jsonl
# data_dir=/home/admin/workspace/aop_lab/q_llm_embedding/datas/hpc_tb_search_queryrec_dichi_diwen_step4_bts_clustering_1020_for_emb_shuf100w_train_tmp_use.jsonl
data_dir=/home/admin/workspace/aop_lab/q_llm_embedding/datas/log_data/hpc_tb_search_queryrec_searchlog_20251020_100w_for_emb_train_tmp_use1.jsonl
num_train_epochs=1
# ds_config=/home/admin/workspace/aop_lab/git_packs/easy_rag/examples/finetune/ds_config.json
ds_config=/home/admin/workspace/aop_lab/git_packs/safetygpt/configs/ds_configs/ds_config_zero1.json
per_device_train_batch_size=8
gradient_accumulation_steps=8
train_group_size=11

# 标准 
# torchrun --nproc_per_node 8 --nnodes 1 \
# /home/admin/workspace/aop_lab/git_packs/easy_rag/FlagEmbedding/baai_general_embedding/finetune/run.py \
# --output_dir $model_output_dir \
# --model_name_or_path $model_path \
# --train_data $data_dir \
# --learning_rate 1e-5 \
# --save_strategy epoch \
# --per_device_train_batch_size $per_device_train_batch_size \
# --gradient_accumulation_steps $gradient_accumulation_steps \
# --bf16 \
# --num_train_epochs $num_train_epochs \
# --dataloader_drop_last True \
# --normlized True \
# --temperature 0.02 \
# --query_max_len 64 \
# --passage_max_len 64 \
# --save_on_each_node False \
# --train_group_size $train_group_size \
# --negatives_cross_device \
# --logging_steps 10 \
# --deepspeed $ds_config

# llm special
cd /home/admin/workspace/aop_lab/git_packs/easy_rag/FlagEmbedding/baai_general_embedding/finetune 
torchrun --nproc_per_node 8 --nnodes 1 \
/home/admin/workspace/aop_lab/git_packs/easy_rag/FlagEmbedding/baai_general_embedding/finetune/run.py \
--output_dir $model_output_dir \
--model_name_or_path $model_path \
--train_data $data_dir \
--learning_rate 1e-5 \
--save_strategy epoch \
--per_device_train_batch_size $per_device_train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps \
--bf16 \
--num_train_epochs $num_train_epochs \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 64 \
--save_on_each_node False \
--train_group_size $train_group_size \
--negatives_cross_device \
--logging_steps 10 \
--llm_head_embedding_mode True \
--llm_embedding_token_type special \
--deepspeed $ds_config

# cd /home/admin/workspace/aop_lab/git_packs/easy_rag/FlagEmbedding/baai_general_embedding/finetune 
# torchrun --nproc_per_node 4 --nnodes 1 \
# /home/admin/workspace/aop_lab/git_packs/easy_rag/FlagEmbedding/baai_general_embedding/finetune/run.py \
# --output_dir $model_output_dir \
# --model_name_or_path $model_path \
# --train_data $data_dir \
# --learning_rate 1e-5 \
# --save_strategy epoch \
# --per_device_train_batch_size $per_device_train_batch_size \
# --bf16 \
# --num_train_epochs $num_train_epochs \
# --dataloader_drop_last True \
# --normlized True \
# --temperature 0.02 \
# --query_max_len 64 \
# --passage_max_len 64 \
# --save_on_each_node False \
# --train_group_size $train_group_size \
# --negatives_cross_device \
# --logging_steps 10 \
# --llm_head_embedding_mode True \
# --llm_embedding_token_type special


# byte
# PORTS=$ARNOLD_WORKER_0_PORT 
# PORT=(${PORTS//,/ }) 

# /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/s
# TORCHRUN --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM \
# --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
# --master_port $PORT \
# run.py \
# --output_dir $model_output_dir \
# --model_name_or_path $model_path \
# --train_data $data_dir \
# --learning_rate 1e-5 \
# --save_strategy epoch \
# --per_device_train_batch_size $per_device_train_batch_size \
# --fp16 \
# --num_train_epochs $num_train_epochs \
# --dataloader_drop_last True \
# --normlized True \
# --temperature 0.02 \
# --query_max_len 64 \
# --passage_max_len 64 \
# --save_on_each_node False \
# --train_group_size $train_group_size \
# --negatives_cross_device \
# --logging_steps 10 
# --deepspeed $ds_config \
# --gradient_checkpointing


# TORCHRUN run.py \
# --output_dir $model_output_dir \
# --model_name_or_path $model_path \
# --train_data $data_dir \
# --learning_rate 1e-5 \
# --save_strategy epoch \
# --fp16 \
# --num_train_epochs $num_train_epochs \
# --per_device_train_batch_size 12 \
# --dataloader_drop_last True \
# --normlized True \
# --temperature 0.02 \
# --query_max_len 64 \
# --passage_max_len 256 \
# --train_group_size 5 \
# --negatives_cross_device \
# --logging_steps 10

# bytenas://cn:zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments
# TORCHRUN run.py \
# --output_dir /mnt/bn/zhangyuntao-workspace-03/wulongyun/bge_baai/output/bge_merlin_60w_1e5_tsg5 \
# --model_name_or_path /mnt/bn/zhangyuntao-workspace-03/wulongyun/bge_baai/bge-large-zh \
# --train_data /mnt/bn/zhangyuntao-workspace-03/wulongyun/bge_baai/data/train_data_mnHN_new_6w.jsonl \
# --learning_rate 1e-5 \
# --save_strategy epoch \
# --fp16 \
# --num_train_epochs 1 \
# --per_device_train_batch_size 12 \
# --dataloader_drop_last True \
# --normlized True \
# --temperature 0.02 \
# --query_max_len 64 \
# --passage_max_len 256 \
# --train_group_size 5 \
# --negatives_cross_device \
# --logging_steps 10