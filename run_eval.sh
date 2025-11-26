

# python reference_imgs_scripts/qwen_generate_multi.py \
#     --node_rank 0 \
#     --num_nodes 1 \
#     --num_variations 8 \
#     --output_dir "" \
#     --text_file ""



python3 inference_t2i.py --config config/grpo.py:eval_sd3_fast --prompts "a flower on a planet"


