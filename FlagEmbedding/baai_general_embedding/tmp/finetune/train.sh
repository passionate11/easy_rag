# # torchrun --nproc_per_node {number of gpus} \
# torchrun --nproc_per_node=2 --nnodes=1 --master_port=29500 \
#     -m FlagEmbedding.baai_general_embedding.finetune.run \
#     --output_dir {path to save model} \
#     --model_name_or_path /data2/yaosijia/MiniCPM-2B \
#     --train_data /data3/yaosijia/Dataset/MS_MARCO/train.new.jsonl \
#     --learning_rate 1e-5 \
#     --fp16 \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size {large batch size; set 1 for toy data} \
#     --dataloader_drop_last True \
#     --normlized True \
#     --temperature 0.02 \
#     --query_max_len 32 \
#     --passage_max_len 128 \
#     --train_group_size 9 \
#     --negatives_cross_device \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --query_instruction_for_retrieval "" 

# formatted_time=$(date +"%Y%m%d%H%M%S")
# echo $formatted_time



# formatted_time=$(date +"%Y%m%d%H%M%S")
# echo $formatted_time

# torchrun --nproc_per_node 2 \
#     -m FlagEmbedding.baai_general_embedding.finetune.run \
#     --output_dir ./MiniCPM_gist_fintune/output/$formatted_time/ \
#     --model_name_or_path /data3/yaosijia/models/MiniCPM-gist \
#     --model_type gist \
#     --train_data /data3/yaosijia/Dataset/MS_MARCO/train.new.jsonl \
#     --learning_rate 1e-5 \
#     --bf16 \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 8 \
#     --dataloader_drop_last True \
#     --normlized True \
#     --temperature 0.02 \
#     --query_max_len 32 \
#     --passage_max_len 128 \
#     --train_group_size 9 \
#     --use_lora False \
#     --negatives_cross_device \
#     --logging_steps 10 \
#     --logging_dir ./MiniCPM_gist_fintune/logs/ \
#     --save_steps 20000 \
   

formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir ./MiniCPM_gist_fintune/output/$formatted_time/ \
    --model_name_or_path /data3/yaosijia/models/MiniCPM-gist \
    --use_model_type gist \
    --train_data /data3/yaosijia/Dataset/MS_MARCO/train.test.jsonl \
    --learning_rate 1e-5 \
    --bf16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 32 \
    --passage_max_len 128 \
    --train_group_size 9 \
    --use_lora False \
    --negatives_cross_device \
    --logging_steps 10 \
    --logging_dir ./MiniCPM_gist_fintune/logs/ \
    --save_steps 20000 \
   