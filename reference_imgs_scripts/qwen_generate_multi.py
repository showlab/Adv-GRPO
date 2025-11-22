

import os
import json
import argparse
import torch
import numpy as np
from diffusers import DiffusionPipeline
from multiprocessing import Process, Queue, set_start_method

# ========== 配置 ==========
# train_file = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/dataset/pickscore/test.txt"
# train_file = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/dataset/geneval/train.txt"
# output_dir = "/mnt/bn/vgfm2/test_dit/weijia/outputs_flowgrpo_test/qwen_images_pickscore_train"

# train_file = ""
# output_dir = ""

os.makedirs(output_dir, exist_ok=True)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for English prompt
    "zh": ", 超清，4K，电影级构图."                  # for Chinese prompt
}
negative_prompt = " "



# ========== worker 函数 ==========
def worker(output_dir, node_rank, rank, prompts, device, q, steps=50, num_variations=8):
    try:
        # 加载模型到指定 GPU
        qwen_pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch.bfloat16
        ).to(device)

        prompt2img_local = {}
        for i, p in enumerate(prompts):
            # 检查是否已经生成过
            existing_imgs = [
                f"node{node_rank}_rank{rank}_{i:05d}_{k}.png"
                for k in range(num_variations)
                if os.path.exists(
                    os.path.join(output_dir, f"node{node_rank}_rank{rank}_{i:05d}_{k}.png")
                )
            ]

            if len(existing_imgs) == num_variations:
                prompt2img_local[p] = existing_imgs
                continue

            # 构造 prompt（重复 num_variations 次，一次 forward）
            batched_prompts = [p] * num_variations
            final_prompts = [
                (pp + positive_magic["en"]) if all(ord(c) < 128 for c in pp)
                else (pp + positive_magic["zh"])
                for pp in batched_prompts
            ]

            images = qwen_pipe(
                prompt=final_prompts,
                negative_prompt=negative_prompt,
                width=512,
                height=512,
                num_inference_steps=steps,
                true_cfg_scale=4.0,
            ).images

            # 保存图片
            img_list = []
            for k, img in enumerate(images):
                img_name = f"node{node_rank}_rank{rank}_{i:05d}_{k}.png"
                img_path = os.path.join(output_dir, img_name)
                img.save(img_path)
                img_list.append(img_name)

            prompt2img_local[p] = img_list

        q.put(prompt2img_local)
    except Exception as e:
        print(f"[Worker {rank}] Error: {e}")
        q.put({})


# ========== 主进程 ==========
if __name__ == "__main__":
    set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--node_rank", type=int, default=0, help="Index of the current node, starting from 0")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--num_variations", type=int, default=8, help="Number of images generated per prompt")
    parser.add_argument("--text_file", type=str, help="Path to the input prompt text file")
    parser.add_argument("--output_dir", type=str, help="Directory to save generated results")

    args = parser.parse_args()

    # 读取 prompts
    with open(args.text_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # 多机拆分
    prompts_per_node = np.array_split(prompts, args.num_nodes)[args.node_rank]

    # 多卡拆分
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    prompts_per_gpu = np.array_split(prompts_per_node, len(devices))

    q = Queue()
    procs = []

    # 启动子进程
    for rank, (dev, sub_prompts) in enumerate(zip(devices, prompts_per_gpu)):
        p = Process(target=worker, args=(args.output_dir, args.node_rank, rank, sub_prompts, dev, q, args.steps, args.num_variations))
        p.start()
        procs.append(p)

    # 收集结果并实时写入
    prompt2img_global = {}
    json_path = os.path.join(args.output_dir, f"prompt2img_node{args.node_rank}.json")

    for _ in procs:
        res = q.get()
        for k, v in res.items():
            if k not in prompt2img_global:
                prompt2img_global[k] = v
            else:
                prompt2img_global[k].extend(v)

    for p in procs:
        p.join()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(prompt2img_global, f, ensure_ascii=False, indent=2)

    print(f"\nNode {args.node_rank} finished! Generated images for {len(prompts_per_node)} prompts. Results have been saved to {json_path}.")

