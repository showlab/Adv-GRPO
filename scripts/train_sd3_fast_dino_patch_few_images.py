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
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
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

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

torch.autograd.set_detect_anomaly(True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

criterion = CLIPCriterion(CLIPCriterionConfig())

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train', limit=None):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            if limit is None:
                self.prompts = [line.strip() for line in f.readlines()]
            else:
                self.prompts = [line.strip() for line in f.readlines()][:limit]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        # import pdb; pdb.set_trace()
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs



def tensor_to_pil_list(tensor_imgs):
    """
    tensor_imgs: torch.Tensor [B,3,H,W], range [-1,1]
    return: list of PIL Images
    """
    imgs = []
    for img in tensor_imgs:
        # [-1,1] -> [0,1]
        img = (img * 0.5 + 0.5).clamp(0, 1)
        # [C,H,W] -> [H,W,C]
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        # [0,1] -> [0,255] uint8
        img = (img * 255).astype(np.uint8)
        # to PIL
        imgs.append(Image.fromarray(img))
    return imgs


def train_dino(scorer, head, prompts, external_imgs_pil, generated_imgs_pil, optimizer_D, accelerator, n_patches=64, patch_loss_weight=0.3):
    """
    单次训练 step，加入 CLS + 随机采样 patch loss
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    device = accelerator.device

    # --- 预处理 ---
    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # 转换 PIL -> tensor
    pos_imgs = torch.stack([transform(img) for img in external_imgs_pil]).to(device).to(torch.bfloat16)
    neg_imgs = torch.stack([transform(img) for img in generated_imgs_pil]).to(device).to(torch.bfloat16)

    # --- forward ---
    scorer.eval()  # backbone 固定
    head.train()   # 训练 head

    with torch.no_grad():
        out_real = scorer.forward_features(pos_imgs)   # [B, N+1, D]
        out_fake = scorer.forward_features(neg_imgs)   # [B, N+1, D]

        cls_real, patch_real = out_real[:,0], out_real[:,1:]   # [B,D], [B,N,D]
        cls_fake, patch_fake = out_fake[:,0], out_fake[:,1:]   # [B,D], [B,N,D]

    # --- CLS image-wise logits ---
    logits_real_cls = head(cls_real).squeeze(-1)   # [B]
    logits_fake_cls = head(cls_fake).squeeze(-1)   # [B]

    # Hinge loss (CLS-level)
    loss_real_cls = torch.mean(F.relu(1.0 - logits_real_cls))
    loss_fake_cls = torch.mean(F.relu(1.0 + logits_fake_cls))
    image_loss = 0.5 * (loss_real_cls + loss_fake_cls)

    # --- Patch-level logits (随机采样 n_patches) ---
    B, N, D = patch_real.shape
    n_select = min(n_patches, N)

    # 随机 index（每张图采样不同的 patch）
    idx_real = torch.randint(0, N, (B, n_select), device=device)
    idx_fake = torch.randint(0, N, (B, n_select), device=device)

    sampled_real = torch.gather(patch_real, 1, idx_real.unsqueeze(-1).expand(-1, -1, D))  # [B,n,D]
    sampled_fake = torch.gather(patch_fake, 1, idx_fake.unsqueeze(-1).expand(-1, -1, D))  # [B,n,D]

    logits_real_patch = head(sampled_real).squeeze(-1)  # [B,n]
    logits_fake_patch = head(sampled_fake).squeeze(-1)  # [B,n]

    # Patch-level hinge loss
    loss_real_patch = torch.mean(F.relu(1.0 - logits_real_patch))
    loss_fake_patch = torch.mean(F.relu(1.0 + logits_fake_patch))
    patch_loss = 0.5 * (loss_real_patch + loss_fake_patch)

    # --- Total loss ---
    d_loss = image_loss + patch_loss_weight * patch_loss
    # d_loss = patch_loss
    # import pdb; pdb.set_trace()

    # --- backward ---
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # --- accuracy (用CLS判断) ---
    preds_real = (logits_real_cls > 0).float()
    preds_fake = (logits_fake_cls < 0).float()
    acc = 0.5 * (preds_real.mean().item() + preds_fake.mean().item())

    return d_loss.item(), acc


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
    
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()


        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters,scorer):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    file_path = os.path.join(config.test_external_image_path,"prompt2img_node0.json")
    with open(file_path, "r", encoding="utf-8") as f:
        external_images_dic = json.load(f)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        # import pdb; pdb.set_trace()
        # prompts = ["an image of a cat on head of trump "]
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # import pdb; pdb.set_trace()
        
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        # import pdb; pdb.set_trace()
        generator = torch.Generator()
        generator.manual_seed(0)
        
        with autocast():
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
                    process_index=accelerator.process_index,
                    sample_num_steps=config.sample.num_steps,
                    random_timestep = config.sample.random_timestep,
                    generator = generator,
                )
        # rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        external_images = []
        for prompt in prompts:
            if prompt in external_images_dic:
                # file_path =  os.path.join(config.test_external_image_path,external_images_dic[prompt][0])
                file_path =  os.path.join(config.test_external_image_path,external_images_dic[prompt])
            else:
                file_path = "/mnt/bn/vgfm2/test_dit/weijia/adv_grpo/img0.png"
                # print("flux does exist the images")
            external_image = Image.open(file_path)
            external_images.append(external_image)
        # import pdb; pdb.set_trace()
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # 转 [0,1]
            # transforms.Normalize([0.5], [0.5])  # 映射到 [-1,1]
        ])

        img_tensors = [preprocess(img) for img in external_images]  # list of [3,512,512]
        img_tensor = torch.stack(img_tensors, dim=0).to(accelerator.device, dtype=torch.float32)  # [B,3,512,512]
        # import pdb; pdb.set_trace()
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, scorer = scorer, only_strict=True,  ref_images = img_tensor)
        # rewards = executor.submit(reward_fn, images.to(torch.bfloat16), prompts, prompt_metadata, scorer = scorer, only_strict=True, ref_images = img_tensor.to(torch.bfloat16))
        # rewards = reward_fn(images.to(torch.bfloat16), prompts, prompt_metadata, scorer = scorer, only_strict=True, ref_images = img_tensor.to(torch.bfloat16))
        # rewards = reward_fn(images, prompts, prompt_metadata, scorer = scorer, only_strict=True, ref_images = img_tensor)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            # rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device, dtype=torch.float32)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    # import pdb; pdb.set_trace()
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device, dtype=torch.float32)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device, dtype=torch.float32)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * config.sample.train_num_steps,
    )
    if accelerator.is_main_process:
        if config.wandb_init:
            wandb.init(
                project="adv_grpo",
                name=f"case_{config.case_name}", 
            )
        else:
            pass
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

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
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

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



    # scorer = PickScoreScorer(dtype=torch.bfloat16, device=accelerator.device)
    scorer = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
    scorer.eval().to(accelerator.device)

    class DINOHead(nn.Module):
        def __init__(self, in_dim=1024, hidden_dim=512):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)  # 输出一个分数
            )

        def forward(self, x):
            return self.layers(x)
    head = DINOHead(in_dim=scorer.num_features).to(accelerator.device)



    head.requires_grad = True

    # weight_path = "/mnt/bn/vgfm2/test_dit/weijia/adv_grpo/stylegan_discriminator.pth"
    weight_path = config.weight_path
    # weight_path = None
    if weight_path is not None:
        state_dict = torch.load(weight_path, map_location="cpu")
        scorer.load_state_dict(state_dict)
        print(f"Loaded discriminator weights from {weight_path}")
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)


    # scorer_trainable_parameters = list(filter(lambda p: p.requires_grad, scorer.parameters()))



    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        # all_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )


    # prepare prompt and reward fn
    reward_fn = getattr(adv_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(adv_grpo.rewards, 'multi_score')(accelerator.device, config.eval_reward_fn)
    # import pdb; pdb.set_trace()

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train', limit=config.limit)
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt//config.sample.mini_num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt//config.sample.mini_num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size*config.sample.mini_num_image_per_prompt, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size*config.sample.mini_num_image_per_prompt, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # import pdb; pdb.set_trace()
    # autocast = accelerator.autocast

    if accelerator.state.deepspeed_plugin:
         accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size
    

    head = head.to(torch.bfloat16)
    scorer = scorer.to(torch.bfloat16)
    scorer.requires_grad = False
    local_rank = accelerator.local_process_index
    head = DDP(head, device_ids=[local_rank], output_device=local_rank)
    optimizer_D = torch.optim.Adam(head.parameters(), lr=config.d_lr, betas=(0.5, 0.999))

    # Prepare everything with our `accelerator`.
    # model, optimizer, optimizer_D, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, optimizer_D, train_dataloader, test_dataloader)
    transformer, optimizer,  train_dataloader, test_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    # file_path = os.path.join(config.external_image_path,"prompt2img.json")
    # file_path = "/mnt/bn/vgfm2/test_dit/weijia/adv_grpo/prompt2img_merged.json"
    file_path = config.json_path
    with open(file_path, "r", encoding="utf-8") as f:
        external_images_dic = json.load(f)
    
    while True:
        # #################### EVAL ####################
        if config.wandb_init:
            pipeline.transformer.eval()
            if (epoch) % config.eval_freq == 0:
                eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, scorer)
            if (epoch) % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
                save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        generated_imgs = []
        external_imgs = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            # import pdb; pdb.set_trace()

            # sample
            with autocast():
                with torch.no_grad():
                    # import pdb; pdb.set_trace()
                    # import time

                    # start = time.time()
                    images, latents, log_probs, timesteps = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        mini_num_image_per_prompt=config.sample.mini_num_image_per_prompt,
                        train_num_steps=config.sample.train_num_steps,
                        process_index=accelerator.process_index,
                        sample_num_steps=config.sample.num_steps,
                        random_timestep = config.sample.random_timestep,
                    )
                    if prompts[0] in external_images_dic:
                        # import pdb; pdb.set_trace()
                        # file_list = []
                        # for i in range(8):
                        #     random_prompt = random.choice(list(external_images_dic.keys()))
                        #     file_list.append(external_images_dic[random_prompt][0])
                        file_list = external_images_dic[prompts[0]][:1]*8  # 这是一个list
                        # file_list = external_images_dic[prompts[0]]  # 这是一个list
                        external_images = []
                        for fname in file_list:
                            fpath = os.path.join(config.external_image_path, fname)
                            try:
                                img = Image.open(fpath).convert("RGB")
                                external_images.append(img)
                            except Exception as e:
                                print(f"[Error] Failed to open {fpath}: {e}")
                                default_path = "/mnt/bn/vgfm2/test_dit/weijia/adv_grpo/img.png"
                                img = Image.open(default_path).convert("RGB")
                                external_images.append(img)
                    else:
                        # 用默认图片兜底
                        print(f"[Warning] external image not found for prompt: {prompts[0]}")
                        default_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_8/node0_rank0_00001_0.png"
                        external_images = [Image.open(default_path).convert("RGB")]*8
                    
                    preprocess = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),  # 转 [0,1]
                        # transforms.Normalize([0.5], [0.5])  # 映射到 [-1,1]
                    ])
                    # 批量处理
                    img_tensors = [preprocess(img) for img in external_images]  # list of [3,512,512]
                    img_tensor = torch.stack(img_tensors, dim=0).to(accelerator.device, dtype=torch.float32)  # [B,3,512,512]
                    generated_imgs.append(images)
                    external_imgs.append(img_tensor)
                    # import pdb; pdb.set_trace()

                    
                    # end = time.time()
                    # print("运行时间: {:.2f} 秒".format(end - start))
                    # print("images dtype:", images[0].dtype)
                    # print("latents dtype:", latents[0].dtype)
                    # print("log_probs dtype:", log_probs[0].dtype)
                    # print("timesteps dtype:", timesteps[0].dtype)
                    # import pdb; pdb.set_trace()


            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            timesteps = torch.stack(timesteps, dim=1)  # shape after stack (batch_size, num_steps)
            # compute rewards asynchronously
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids.repeat(config.sample.mini_num_image_per_prompt,1), skip_special_tokens=True
            )

            rewards = executor.submit(reward_fn, images.to(torch.bfloat16), prompts, prompt_metadata, scorer = scorer, head = head,  only_strict=True)
            external_rewards = executor.submit(reward_fn, img_tensor.to(torch.bfloat16), prompts, prompt_metadata, scorer = scorer,  head = head,  only_strict=True)
            # import pdb; pdb.set_trace()
            # rewards = reward_fn(images.to(torch.bfloat16), prompts, prompt_metadata, scorer = scorer,head = head,  only_strict=True)
            # import pdb; pdb.set_trace()
            # yield to to make sure reward computation starts
            time.sleep(0)
            samples.append(
                {
                    "prompt_ids": prompt_ids.repeat(config.sample.mini_num_image_per_prompt,1),
                    "prompt_embeds": prompt_embeds.repeat(config.sample.mini_num_image_per_prompt,1,1),
                    "pooled_prompt_embeds": pooled_prompt_embeds.repeat(config.sample.mini_num_image_per_prompt,1),
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "external_rewards": external_rewards,
                }
            )


        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            external_rewards, external_reward_metadata = sample["external_rewards"].result()
            # accelerator.print(reward_metadata)
            # import pdb; pdb.set_trace()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            sample["external_rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in external_rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引
                    # pil.save( f"{idx}.jpg")
                # import pdb; pdb.set_trace()
                # print("prompts",len(prompts))

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]
                if config.wandb_init:
                    wandb.log(
                        {
                            "images": [
                                wandb.Image(
                                    os.path.join(tmpdir, f"{idx}.jpg"),
                                    caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                                )
                                for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                            ],
                        },
                        step=global_step,
                    )
            # import pdb; pdb.set_trace()
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(img_tensor))
                sample_indices = random.sample(range(len(img_tensor)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = img_tensor[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引
                if config.wandb_init:
                    wandb.log(
                        {
                            "external_images": [
                                wandb.Image(
                                    os.path.join(tmpdir, f"{idx}.jpg"),
                                )
                                for  idx, i in enumerate(sample_indices)
                            ],
                        },
                        step=global_step,
                    )

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, config.sample.train_num_steps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        samples["external_rewards"]["ori_avg"] = samples["external_rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["external_rewards"]["avg"] = samples["external_rewards"]["avg"].unsqueeze(1).repeat(1, config.sample.train_num_steps)
        # gather rewards across processes
        gathered_external_rewards = {key: accelerator.gather(value) for key, value in samples["external_rewards"].items()}
        gathered_external_rewards = {key: value.cpu().numpy() for key, value in gathered_external_rewards.items()}

        # log rewards and images
        if accelerator.is_main_process and config.wandb_init:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"external_reward_{key}": value.mean() for key, value in gathered_external_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids_per_gpu = samples["prompt_ids"].cpu().numpy()
            prompts_per_gpu = pipeline.tokenizer.batch_decode(
                prompt_ids_per_gpu, skip_special_tokens=True
            )
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process and config.wandb_init:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)
        # import pdb; pdb.set_trace()

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
        
        external_imgs = torch.cat(external_imgs, dim=0)  # [N, 3, 512, 512]
        generated_imgs = torch.cat(generated_imgs, dim=0)  # [N, 3, 512, 512]

        external_imgs = (external_imgs - 0.5) * 2.0
        generated_imgs = (generated_imgs - 0.5) * 2.0
        rewards_mean = accelerator.gather(samples["rewards"]['dino_patch_cotrain']).mean()
        external_rewards_mean = accelerator.gather(samples["external_rewards"]['dino_patch_cotrain']).mean()

        # rewards
        head.requires_grad = True
        transformer.requires_grad = False
        # if config.train_d:
        if config.train_d and (epoch+1)%config.d_times!=0:
        # if config.train_d and external_rewards_mean < rewards_mean:
            # import pdb; pdb.set_trace()
            external_imgs_pil = tensor_to_pil_list(external_imgs)
            generated_imgs_pil = tensor_to_pil_list(generated_imgs)
            # import pdb; pdb.set_trace()
            len_ext = len(external_imgs_pil)
            len_gen = len(generated_imgs_pil)

            if len_ext != len_gen:
                min_len = min(len_ext, len_gen)
                print(f"[Warning] external_imgs_pil({len_ext}) and generated_imgs_pil({len_gen}) length mismatch, truncating to {min_len}")
                external_imgs_pil = external_imgs_pil[:min_len]
                generated_imgs_pil = generated_imgs_pil[:min_len]
                
            d_loss ,acc = train_dino(scorer, head, prompts_per_gpu, external_imgs_pil, generated_imgs_pil, optimizer_D, accelerator)

            # import pdb; pdb.set_trace()
            if accelerator.is_main_process and config.wandb_init:
                wandb.log({"train/d_loss": d_loss}, step=global_step)
                wandb.log({"train/acc": acc}, step=global_step)
            global_step+=1
            epoch+=1
            continue

        # if config.train_d and external_rewards_mean < rewards_mean:
        #     ext_scores = samples["external_rewards"]['pickscore_cotrain']# [B]
        #     gen_scores = samples["rewards"]['pickscore_cotrain']# [B]
        #     mask = gen_scores > ext_scores  # [B] bool 张量
        #     external_imgs_pil = tensor_to_pil_list(external_imgs)
        #     generated_imgs_pil = tensor_to_pil_list(generated_imgs)
        #     filtered_ext_imgs = []
        #     filtered_gen_imgs = []
        #     filtered_prompts = []
        #  
        # import pdb; pdb.set_trace()




        head.requires_grad = False
        transformer.requires_grad = True

        del samples["rewards"]
        del samples["external_rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )





        #################### TRAINING ####################
        

        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                for j in tqdm(
                    range(config.sample.train_num_steps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            # kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) 
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        # info["discriminator_loss"].append(d_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        # import pdb; pdb.set_trace()
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

