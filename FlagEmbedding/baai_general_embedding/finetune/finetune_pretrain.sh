# 模型路径 模型输出路径 数据路径 epoch数量
model_path=$1
model_output_dir=$2
# /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/data/contravasive_data/bge_train_data_new.jsonl
# /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/data/contravasive_data/bge_train_data_new_minedHN.jsonl
data_dir=$3
num_train_epochs=$4
ds_config=$5

PORTS=$ARNOLD_WORKER_0_PORT 
PORT=(${PORTS//,/ }) 

# /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/s
torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM \
--node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
--master_port $PORT \
run.py \
--output_dir $model_output_dir --deepspeed $ds_config --num_train_epochs $num_train_epochs \
--model_name_or_path $model_path \
--train_data $data_dir \
--learning_rate 1e-5 \
--save_strategy epoch \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 1600 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval ""    \


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