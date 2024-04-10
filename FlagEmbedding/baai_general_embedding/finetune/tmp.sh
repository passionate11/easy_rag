sudo chown -R tiger /mnt/bn/zhangyuntao-workspace-03/hpc
sudo pip3 install -U FlagEmbedding
sudo pip3 install byted-cruise==0.7.0
sudo pip3 install --upgrade requests urllib3 chardet
cd /opt/tiger/FlagEmbedding/FlagEmbedding/baai_general_embedding/finetune/
torchrun --nnodes $ARNOLD_WORKER_NUM  --nproc_per_node 8 --master_port 7129 run.py --learning_rate 1e-5 --save_strategy epoch --dataloader_drop_last True --normlized True --temperature 0.02 --query_max_len 512 --passage_max_len 512 --negatives_cross_device --logging_steps 10 --train_group_size 10 --query_instruction_for_retrieval "" --model_name_or_path /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/mae_bge_large/version_3/encoder_model --output_dir /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/finetuned_model/my_large_mae_cl_gradient_tmp/checkpoints   --train_data /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/data/fintune_data_version_0_pre1000.jsonl --num_train_epochs 2  --per_device_train_batch_size 20 --deepspeed /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/ds_config/example.json --gradient_checkpointing


torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST --master_port $PORT run.py --learning_rate 1e-5 --save_strategy epoch --dataloader_drop_last True --normlized True --temperature 0.02 --query_max_len 512 --passage_max_len 512 --negatives_cross_device --logging_steps 10 --train_group_size 10 --query_instruction_for_retrieval "" --model_name_or_path /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/mae_bge_large/version_3/encoder_model --output_dir /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/finetuned_model/my_large_mae_cl_gradient_tmp/checkpoints   --train_data /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/data/fintune_data_version_0_pre1000.jsonl --num_train_epochs 5  --per_device_train_batch_size 20 --deepspeed /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/ds_config/example.json --gradient_checkpointing


torchrun  --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST --master_port $PORT run.py --learning_rate 1e-5 --save_strategy epoch --dataloader_drop_last True --normlized True --temperature 0.02 --fp16 --query_max_len 512 --passage_max_len 512 --negatives_cross_device --logging_steps 10 --train_group_size 10 --query_instruction_for_retrieval "" --model_name_or_path /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/mae_bge_large/version_3/encoder_model --output_dir /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/finetuned_model/my_large_mae_cl_gradient_tmp/checkpoints   --train_data /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/data/fintune_data_version_0_pre1000.jsonl --num_train_epochs 2  --per_device_train_batch_size 20 --deepspeed /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/ds_config/example.json --gradient_checkpointing 


cd /opt/tiger/FlagEmbedding/FlagEmbedding/baai_general_embedding/finetune/
torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST --master_port $PORT run.py --learning_rate 1e-5 --save_strategy steps --dataloader_drop_last True --normlized True --temperature 0.02 --query_max_len 64 --passage_max_len 512 --negatives_cross_device --logging_steps 10 --train_group_size 11 \
--model_name_or_path /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/finetuned_model/my_large_mae_cl_gradient_32_ka/checkpoints/checkpoint-1215 \
--output_dir /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/3.20_my_bge_large_d5_inbatch_neg_modified_loss_debug_use \
--train_data /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/3.19_xiangqian_bge_base_d5/3.19_all_train_data_pre1000.jsonl \
--save_steps 1 \
--num_train_epochs 1 \
--per_device_train_batch_size 40 \
--deepspeed /mnt/bn/zhangyuntao-workspace-03/hpc/ecom_embedding_for_refusal/experiments/Embedding_models/ds_config/example.json \
--gradient_checkpointing --fp16 --use_inbatch_neg --use_my_modified_loss_model

	
sudo chown -R tiger /mnt/bn/zhangyuntao-workspace-03/hpc
sudo pip3 install -U FlagEmbedding
sudo pip3 install byted-cruise==0.7.0
sudo pip3 install --upgrade transformers
sudo pip3 install --upgrade requests urllib3 chardet
cd /opt/tiger/FlagEmbedding/FlagEmbedding/baai_general_embedding/finetune/