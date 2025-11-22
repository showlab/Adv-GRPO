

import os
import json
import torch
import numpy as np
from diffusers import DiffusionPipeline
from multiprocessing import Process, Queue, set_start_method

# ========== 配置 ==========
train_file = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/dataset/ocr/train.txt"
output_dir = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_8"
os.makedirs(output_dir, exist_ok=True)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for English prompt
    "zh": ", 超清，4K，电影级构图."                  # for Chinese prompt
}
negative_prompt = " "


# 每个 GPU 的随机数生成器，保证不同 GPU 上也可复现
def get_generator(device, seed=42):
    return torch.Generator(device=device).manual_seed(seed)


# ========== worker 函数 ==========
def worker(rank, prompts, device, q, steps=50, batch_size=4):
    # 加载 Qwen 模型到指定 GPU
    qwen_pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=torch.bfloat16
    ).to(device)

    # generator = get_generator(device, seed=42 + rank)

    prompt2img_local = {}
    for i in range(0, len(prompts), batch_size):
        sub_prompts = prompts[i:i + batch_size]

        # 构建文件名和路径
        img_names = [f"rank{rank}_{i+j:05d}.png" for j in range(len(sub_prompts))]
        img_paths = [os.path.join(output_dir, n) for n in img_names]

        # 记录哪些需要生成
        need_generate = []
        need_prompts = []
        for p, path, name in zip(sub_prompts, img_paths, img_names):
            if os.path.exists(path):
                # print("ok")
                print("exists:"+path)
                # 已存在，直接记录
                prompt2img_local[p] = name
            else:
                need_generate.append((p, path, name))
                need_prompts.append(p)

        # 如果全都存在就跳过
        if not need_prompts:
            continue
        
        final_prompts = [
            (p + positive_magic["en"]) if all(ord(c) < 128 for c in p)
            else (p + positive_magic["zh"])
            for p in need_prompts
        ]

        # 生成图像
        images = qwen_pipe(
            prompt=final_prompts,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            num_inference_steps=steps,
            true_cfg_scale=4.0,
            # generator=generator
        ).images

        # 文件名（加 rank 前缀避免冲突，例如 rank0_00001.png）
        # img_name = f"rank{rank}_{idx:05d}.png"
        # img_path = os.path.join(output_dir, img_name)

        # image.save(img_path)
        # prompt2img_local[prompt] = img_name
        for j, (p, path, name, img) in enumerate(zip(need_prompts, img_paths, img_names, images)):
            img.save(path)
            prompt2img_local[p] = name

        if (i + 1) % 50 == 0:
            print(f"[{device}] 已完成 {idx+1}/{len(prompts)}")

    # 将结果放进队列
    q.put(prompt2img_local)


# ========== 主进程 ==========
if __name__ == "__main__":
    set_start_method("spawn", force=True)

    # 读取 prompts
    with open(train_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    num_gpus = len(devices)
    prompts_per_gpu = np.array_split(prompts, num_gpus)

    q = Queue()
    procs = []

    # 启动子进程
    for rank, (dev, sub_prompts) in enumerate(zip(devices, prompts_per_gpu)):
        p = Process(target=worker, args=(rank, sub_prompts, dev, q, 50, 4))
        p.start()
        procs.append(p)

    # 收集结果
    prompt2img_global = {}
    for _ in procs:
        res = q.get()
        prompt2img_global.update(res)

    for p in procs:
        p.join()

    # 保存映射文件
    json_path = os.path.join(output_dir, "prompt2img.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(prompt2img_global, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共生成 {len(prompts)} 张图像，结果已保存到 {json_path}")
