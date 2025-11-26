<h1 align="center"> The Image as Its Own Reward: <br>
Reinforcement Learning with Adversarial Reward for Image Generation


 </h1>

<div align="center" style="font-size:14px; line-height:1.4; margin-top:10px;">

[Weijia Mao](https://scholar.google.com/citations?user=S7bGBmkyNtEC&hl=zh-CN)<sup>1</sup> 
[Hao Chen](https://haochen-rye.github.io/)<sup>2</sup><sup>✉</sup> 
[Zhenheng Yang](https://zhenheny.github.io/)<sup>2</sup> 
[Mike Zheng Shou](https://sites.google.com/view/showlab/home?authuser=0)<sup>1</sup><sup>✉</sup>  


<sup>1</sup> Show Lab, National University of Singapore,  <sup>2</sup> ByteDance  

</div>
 
<div align="center">
  <a href='https://arxiv.org/abs/2511.20256'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://showlab.github.io/Adv-GRPO/'><img src='https://img.shields.io/badge/Visualization-green?logo=github'></a> &nbsp;
  <a href=""><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/benzweijia/Adv-GRPO'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <!-- <a href='https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO'><img src='https://img.shields.io/badge/Demo-blue?logo=huggingface'></a> &nbsp; -->
  <a href='https://showlab.github.io/Adv-GRPO/assets/Adv-GRPO.pdf'><img src='https://img.shields.io/badge/PDF-orange?logo=adobeacrobatreader'></a> &nbsp;

</div>



## Adv-GRPO

<p align="center">
  <img src="assets/teaser.png" alt="Adv-GRPO Illustration" width=950"/>
</p>

We introduce **Adv-GRPO**, an RL framework with an adversarial reward that iteratively updates
both the reward model and the generator. Our method Adv-GRPO improves text-to-image (T2I) generation in three ways:

 1) Alleviate Reward Hacking, achieving higher perceptual quality while maintaining comparable benchmark performance
(e.g., PickScore, OCR), as shown in the top-left human evaluation panel;

 2) Visual Foundation Model as Reward,
leveraging visual foundation models (e.g., DINO) for rich visual priors, leading to overall improvements as shown
in middle-top human evaluation results; 

 3) RL-based Distribution Transfer, enabling style customization by aligning
generations with reference domains

## Changelog

**2025-11-25**

* We release the code of Adv-GRPO training code, inference code and the pretrained ckpt.

<!-- 
## FAQ -->


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
cd Adv-GRPO
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



And if you do not want to generate, we will provide our reference datasets recently.


<!-- [🤗Reference Dataset](https://huggingface.co/datasets/benzweijia/QWen_Image_PickScore) | -->


Some tips:

- Our reference dataset is relatively large — the full set is about 50 GB if you choose to download it.

- In practice, we do not use all images during training. Similarly, not all prompts are covered when using DINO prompts.

- Based on our ablation studies, using a smaller subset of reference images and prompts can still achieve comparable performance to using the full dataset.

- If you prefer not to use our dataset or have a better alternative, you can use your own dataset and simply adapt it to the required format.



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
# sd3 grpo with DINO reward
bash scripts/multi_node/sd3_fast/grpo_dino.sh
```


```bash
# sd3 grpo with PickScore reward
bash scripts/multi_node/sd3_fast/grpo_pickscore.sh

```

## 📝 TO DO
- Release the reference dataset used in our work  
- Release the DINO reward checkpoint trained with GenEval and OCR prompts  
- Release the style transfer checkpoint  



        
## ✨ Important Tips
1. You can adjust the parameters in `config/grpo.py` to tune different hyperparameters. 


## 🤗 Acknowledgement
This repo is based on [Flow-GRPO](https://github.com/yifan123/flow_grpo.git) . We thank the authors for their valuable contributions to the AIGC community.

<!-- ## ⭐Citation
If you find Adv-GRPO useful for your research or projects, we would greatly appreciate it if you could cite the following paper:

```



``` -->