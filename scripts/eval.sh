
MASTER_PORT=19001
RANK=0
MASTER_ADDR=127.0.0.1
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
--num_machines 1 --num_processes 8 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/eval.py \
    --config config/grpo.py:eval_sd3_fast


