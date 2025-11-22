


#


# python scripts_test/qwen_generate_multi.py --node_rank 0 --num_nodes 1 --num_variations 8


# python3 test2.py



# torchrun --nproc_per_node=8 --master_port=2600 scripts_stylegan/train_dino.py \
#     --external_dir /mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images \
#     --generated_dir /mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images_original_pickscore \
#     --external_val_dir /mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr \
#     --generated_val_dir /mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images_original_ocr \
#     --image_size 518 \
#     --batch_size 32 \
#     --num_epochs 10 \
#     --lr 1e-4 \
#     --output dino_head.pth



# python scripts_test/fid.py --sd3_folder /mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images_original_test_results --qwen_folder /mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test --device cuda

# python3 scripts_test/sd3_generate_multi.py \
#     --train_file /mnt/bn/vgfm2/test_dit/weijia/flow_grpo/dataset/pickscore/test.txt \
#     --output_dir /mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images_dinov2_test_results3 \
#     --lora_path  /mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/discriminator_again/sd3.5-M-fast_dino_cotrain_16_8_lr_times_5_2e5/checkpoints/checkpoint-1056/lora

# torchrun --nproc_per_node=8 --master_port=2600 scripts_test/fid.py \
#   --sd3_folder /mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images_original_test_results_8_dino_similarity \
#   --qwen_folder /mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test_8 \
#   --device cuda


python reference_imgs_scripts/qwen_generate_multi.py \
    --node_rank 0 \
    --num_nodes 1 \
    --num_variations 8 \
    --output_dir "" \
    --text_file ""


