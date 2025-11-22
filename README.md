<h1 align="center"> The Image as Its Own Reward: <br>
Reinforcement Learning with Adversarial Reward for Image Generation </h1>
<div align="center">
  <a href='https://arxiv.org/abs/2505.05470'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://gongyeliu.github.io/Flow-GRPO/'><img src='https://img.shields.io/badge/Visualization-green?logo=github'></a> &nbsp;
  <a href="https://github.com/yifan123/flow_grpo"><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/collections/jieliu/sd35m-flowgrpo-68298ec27a27af64b0654120'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO'><img src='https://img.shields.io/badge/Demo-blue?logo=huggingface'></a> &nbsp;
</div>



## Adv-GRPO

<p align="center">
  <img src="assets/teaser.png" alt="Flow-GRPO-Fast Illustration" width=950"/>
</p>


We introduce **Adv-GRPO**, an RL framework with an adversarial reward that iteratively updates
both the reward model and the generator. The reward model is supervised using reference images
as positive samples and can largely avoid being hacked. Unlike KL regularization that constrains
parameter updates, our learned reward directly guides the generator through its visual outputs,
leading to higher-quality images. Moreover, while optimizing existing reward functions can alleviate
reward hacking, their inherent biases remain. For instance, PickScore may degrade image quality,
whereas OCR-based rewards often reduce aesthetic fidelity. To address this, we take the image itself
as a reward, using reference images and vision foundation models (e.g., DINO) to provide rich visual
rewards. These dense visual signals, instead of a single scalar, lead to consistent gains across image
quality, aesthetics, and task-specific metrics. Finally, we show that combining reference samples with
foundation-model rewards enables distribution transfer and flexible style customization. In human
evaluation, our method outperforms Flow-GRPO and SD3, achieving 70.0% and 72.4% win rates in
image quality and aesthetics, respectively.

## Changelog

**2025-11-22**

* We release the code of Adv-GRPO training code, inference code and the pretrained ckpt.


## FAQ


Please use scripts in `scripts/multi_node/sd3_fast` to run these experiments.

## 🤗 Model
| Task    | Model |
| -------- | -------- |
| PickScore     | [🤗PickScore](https://huggingface.co/benzweijia/Adv-GRPO/tree/main/PickScore) |
| DINOv2     | [🤗DINOv2](https://huggingface.co/benzweijia/Adv-GRPO/tree/main/DINO) |


## 🚀 Quick Started
### 1. Environment Set Up

Clone this repository and install packages.

```bash
git clone https://github.com/showlab/Adv-GRPO.git
cd flow_grpo
conda create -n adv_grpo python=3.10.16 -y
pip install -e .
```

### 2. Reference Image Generation

We use the Qwen-Image model (https://github.com/QwenLM/Qwen-Image) to generate reference images.

First, install the dependencies required by Qwen-Image.

```bash

python reference_imgs_scripts/qwen_generate_multi.py \
    --node_rank 0 \
    --num_nodes 1 \
    --num_variations 8 \
    --output_dir "" \
    --text_file ""
```

The reference images will be saved in output_dir and the json file will be like this:

```json
{
  "middle-aged man with a beard giving a thumbs up, upper body, green fields in the background": [
    "node0_rank3_00000_0.png",
    "node0_rank3_00000_1.png",
    ...
  ],
  "king charles spaniel with planets for eyes, ethereal, midjourney style lighting and shadows, insanely detailed, 8k, photorealistic": [
    "node0_rank3_00001_0.png",
    "node0_rank3_00001_1.png",
    ...
  ],
  ...

}
```


### 2. Inferece Stage.

Firstly, we set the config file  .config/grpo.py


``` python
def eval_sd3_fast():
    ...
     config.train.lora_path = ""
    config.save_folder = ""
    config.json_path = ""
    config.external_image_path = ""
    config.test_external_image_path = ""
    ...
```


Secondly, 

```bash
bash scripts/multi_node/sd3_fast/eval.sh 
```

### 4. Start Training

The config file is in the .config/grpo.py


``` python
def dino_cotrain_sd3_patch_fast():
    ...
    config.json_path = ""
    config.external_image_path = ""
    config.test_external_image_path = ""
    ...
```

We use deepspeed stage2 to save the memory.

```bash
# zero2
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml
# zero3
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero3.yaml
```

#### GRPO
Single-node training:
```bash
# sd3
bash scripts/multi_node/sd3_fast/main.sh
```




        
## ✨ Important Hyperparameters
You can adjust the parameters in `config/grpo.py` to tune different hyperparameters. An empirical finding is that `config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48`, i.e., `group_number=48`, `group_size=24`.
Additionally, setting `config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2`.

## 🤗 Acknowledgement
This repo is based on [Flow-GRPO](https://github.com/yifan123/flow_grpo.git) . We thank the authors for their valuable contributions to the AIGC community.

## ⭐Citation
If you find Adv-GRPO useful for your research or projects, we would greatly appreciate it if you could cite the following paper:
```

```