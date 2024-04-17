py_path=/mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/code_base/ecom_Rag/rag_scripts/2_reranker.py


# python3 $py_path \
# --model_path your_model_path \
# --model_type bge \
# --batch_size 8192 \
# --topk 20 \
# --save_inter_res True \
# --test_data your_to_retriev_data \
# --seed_data your_docs \
# --out_score_path output_sim_path

python3 $py_path \
--model_path $1 \
--model_type $2 \
--batch_size $3 \
--topk $4 \
--in_to_rerank_data_path $5 \
--out_reranked_path $6 