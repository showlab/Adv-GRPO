from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
# from accelerate.utils import set_seed, ProjectConfiguration
# from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import adv_grpo.prompts
import adv_grpo.rewards
from adv_grpo.stat_tracking import PerPromptStatTracker
from adv_grpo.diffusers_patch.sd3_pipeline_with_logprob_fast import pipeline_with_logprob_random as pipeline_with_logprob
from adv_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob_new as sde_step_with_logprob
from adv_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from adv_grpo.ema import EMAModuleWrapper

from torchvision import transforms
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from adv_grpo.pickscore_scorer import PickScoreScorer
from adv_grpo.pick_score_training import CLIPCriterionConfig, CLIPCriterion
import timm
import itertools
from scipy import linalg

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
flags.DEFINE_string("prompts", "", "Prompt text input.")
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")



criterion = CLIPCriterion(CLIPCriterionConfig())




def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def eval(pipeline, text_encoders, tokenizers, config, global_step, executor, num_train_timesteps, ema, transformer_trainable_parameters, prompts):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device="cuda")

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size*8, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size*8, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    idx = 0
    os.makedirs(config.save_folder, exist_ok=True)
    prompt2files_local = {}
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        prompts, 
        text_encoders, 
        tokenizers, 
        max_sequence_length=128, 
        device="cuda"
    )
    # import pdb; pdb.set_trace()
    
    # The last batch may not be full batch_size
    if len(prompt_embeds)<len(sample_neg_prompt_embeds):
        sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
        sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
    # import pdb; pdb.set_trace()
    generator = torch.Generator()
    generator.manual_seed(0)
    
    # with autocast():
    with torch.no_grad():
        images, _, _, _ = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
            num_inference_steps=config.sample.eval_num_steps,
            guidance_scale=config.sample.guidance_scale,
            output_type="pt",
            height=config.resolution,
            width=config.resolution, 
            noise_level=0,
            mini_num_image_per_prompt=1,
            process_index=0,
            sample_num_steps=config.sample.num_steps,
            random_timestep = config.sample.random_timestep,
            generator = generator,
        )
        # rank = accelerator.process_index
        rank = 0
        node_id = 0  # 如果多 node 可以自己传进来
        file_list = []  
        for img_idx, image in enumerate(images):
            pil = Image.fromarray(
                (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            pil = pil.resize((512, 512))

            filename = f"node{node_id}_rank{rank}_{idx:05d}_{img_idx}.png"
            pil.save(os.path.join(filename))
            prompt2files_local[prompts[img_idx]] = [filename]



def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    prompts =  [FLAGS.prompts]

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # import pdb; pdb.set_trace()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=False,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to("cuda", dtype=torch.float32)
    pipeline.text_encoder.to("cuda", dtype=inference_dtype)
    pipeline.text_encoder_2.to("cuda", dtype=inference_dtype)
    pipeline.text_encoder_3.to("cuda", dtype=inference_dtype)
    
    pipeline.transformer.to("cuda")

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device="cuda")


    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    # autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    executor = futures.ThreadPoolExecutor(max_workers=8)

    epoch = 0
    global_step = 0
    pipeline.transformer.eval()
    eval(pipeline, text_encoders, tokenizers, config, global_step, executor, num_train_timesteps, ema, transformer_trainable_parameters, prompts)
        
        
if __name__ == "__main__":
    app.run(main)

