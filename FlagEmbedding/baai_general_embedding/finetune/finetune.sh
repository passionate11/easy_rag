# 模型路径 模型输出路径 数据路径 epoch数量
model_path=/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/mae_bge_large/version_3/encoder_model
model_output_dir=/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/finetuned_model/my_large_mae_cl_gradient/checkpoints_tmp
data_dir=/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/data/fintune_data_version_0_pre1000.jsonl
num_train_epochs=1
ds_config=/mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/ds_config/example.json
per_device_train_batch_size=20
train_group_size=5
 
PORTS=$ARNOLD_WORKER_0_PORT 
PORT=(${PORTS//,/ }) 

# /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/s
TORCHRUN --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM \
--node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
--master_port $PORT \
run.py \
--output_dir $model_output_dir \
--model_name_or_path $model_path \
--train_data $data_dir \
--learning_rate 1e-5 \
--save_strategy epoch \
--per_device_train_batch_size $per_device_train_batch_size \
--fp16 \
--num_train_epochs $num_train_epochs \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 64 \
--save_on_each_node False \
--train_group_size $train_group_size \
--negatives_cross_device \
--logging_steps 10 
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