py_path=/mnt/bn/shuaizzz/hpc/ecom_embedding_for_refusal/scipts/benchmark_scripts/1_scorer.py

python3 $py_path \
--model_path your_model_path \
--model_type bge \
--batch_size 8192 \
--topk 20 \
--save_inter_res True \
--test_data your_to_retriev_data \
--seed_data your_docs \
--out_score_path output_sim_path
