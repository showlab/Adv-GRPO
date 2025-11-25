#!/bin/bash
# Common part for all nodes
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5
# export NCCL_DEBUG=WARN
# export NCCL_IB_GID_INDEX=3


# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=fdbd:dccd:cdc2:12c8:0:3d6::
# # Launch command (parameters automatically read from accelerate_multi_node.yaml)
# accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
#     --num_machines 4 --num_processes 32 \
#     --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#     scripts/train_sd3_fast.py \
#     --config config/grpo.py:pickscore_sd3_fast



# !/bin/bash
# Common part for all nodes
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5
# export NCCL_DEBUG=WARN
# export NCCL_IB_GID_INDEX=3
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
# # 

# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=127.0.0.1
# # Launch command (parameters automatically read from accelerate_multi_node.yaml)
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_machines 1 --num_processes 8 \
#     --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#     scripts/train_sd3_fast_bf.py \
#     --config config/grpo.py:pickscore_sd3_fast





MASTER_PORT=19001
RANK=0
MASTER_ADDR=127.0.0.1
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_machines 1 --num_processes 8 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_sd3_fast_pickscore.py \
    --config config/grpo.py:pickscore_cotrain_sd3_fast




# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=127.0.0.1
# # Launch command (parameters automatically read from accelerate_multi_node.yaml)
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_machines 1 --num_processes 8 \
#     --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#     scripts/train_sd3_fast_dino.py \
#     --config config/grpo.py:dino_cotrain_sd3_fast




# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=127.0.0.1
# # Launch command (parameters automatically read from accelerate_multi_node.yaml)
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_machines 1 --num_processes 8 \
#     --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#     scripts/train_sd3_fast_dino_patch.py \
#     --config config/grpo.py:dino_cotrain_sd3_patch_fast







# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=127.0.0.1
# # Launch command (parameters automatically read from accelerate_multi_node.yaml)
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_machines 1 --num_processes 8 \
#     --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#     scripts/train_sd3_fast_dino_patch.py \
#     --config config/grpo.py:dino_cotrain_sd3_patch_style_fast




# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=127.0.0.1
# # Launch command (parameters automatically read from accelerate_multi_node.yaml)
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_machines 1 --num_processes 1 \
#     --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#     scripts/train_sd3_fast_dino_multi.py \
#     --config config/grpo.py:dino_cotrain_sd3_multi_fast












