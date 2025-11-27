import gradio as gr
import spaces
import torch
from diffusers import StableDiffusion3Pipeline
from adv_grpo.diffusers_patch.sd3_pipeline_with_logprob_fast import pipeline_with_logprob_random as pipeline_with_logprob
from adv_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from adv_grpo.ema import EMAModuleWrapper
from peft import PeftModel
from PIL import Image
import numpy as np
import os
from ml_collections import config_flags
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------
#               GLOBAL VARIABLES
# ---------------------------------------------------------

pipeline = None
config = None
text_encoders = None
tokenizers = None
ema = None
transformer_trainable_parameters = None

def load_lora_from_subfolder():
    repo_id = "benzweijia/Adv-GRPO"
    subfolder = "DINO"          # 

    local_dir = "/tmp/DINO"
    os.makedirs(local_dir, exist_ok=True)

    for filename in ["adapter_config.json", "adapter_model.safetensors"]:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            subfolder=subfolder,
            filename=filename,
            local_dir=local_dir,
            force_download=False
        )
    # import pdb; pdb.set_trace()
    return local_dir

# -------------- Load Config ------------------------------
def load_config():
    """
    直接加载你原先的 config/base.py 配置
    """
    import importlib.util

    config_path = "config/base.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


# -------------- Embedding Function -----------------------
def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------
#              GPU MODEL INITIALIZATION
# ---------------------------------------------------------
@spaces.GPU
def init_model():
    global pipeline, config, text_encoders, tokenizers, ema, transformer_trainable_parameters

    print("🔥 Loading config...")
    config = load_config()

    print("🔥 Loading SD3 base model on GPU...")
    # import pdb; pdb.set_trace()
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium"
    )

    # freeze non-trainable params
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)

    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)

    # move to GPU
    pipeline.vae.to("cuda")
    pipeline.text_encoder.to("cuda")
    pipeline.text_encoder_2.to("cuda")
    pipeline.text_encoder_3.to("cuda")
    pipeline.transformer.to("cuda")
    config.train.lora_path = "benzweijia/Adv-GRPO/DINO"
    config.use_lora = True
    lora_dir = load_lora_from_subfolder()

    if config.use_lora and config.train.lora_path:
        print("🔥 Loading LoRA from:", config.train.lora_path)
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer,
            os.path.join(lora_dir,"DINO")
        )
        pipeline.transformer.set_adapter("default")

    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, pipeline.transformer.parameters())
    )

    # Setup EMA
    ema = EMAModuleWrapper(
        transformer_trainable_parameters,
        decay=0.9,
        update_step_interval=8,
        device="cuda"
    )

    print("✅ Model initialized and ready.")


# ---------------------------------------------------------
#                   INFERENCE FUNCTION
# ---------------------------------------------------------
@spaces.GPU
def infer(prompt):
    global pipeline, config

    if pipeline is None:
        init_model()
        # return "❌ Model not loaded. Something is wrong."

    prompts = [prompt]

    # get prompt embedding
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        prompts, text_encoders, tokenizers,
        max_sequence_length=128,
        device="cuda"
    )

    neg_embed, neg_pooled_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers,
        max_sequence_length=128,
        device="cuda"
    )

    neg_prompt_embeds = neg_embed.repeat(1, 1, 1)
    neg_pooled_prompt_embeds = neg_pooled_embed.repeat(1, 1)

    # generation seed
    generator = torch.Generator().manual_seed(0)

    with torch.no_grad():
        images, _, _, _ = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
            num_inference_steps=config.sample.eval_num_steps,
            guidance_scale=config.sample.guidance_scale,
            output_type="pt",
            height=config.resolution,
            width=config.resolution,
            noise_level=0,
            mini_num_image_per_prompt=1,
            process_index=0,
            sample_num_steps=config.sample.num_steps,
            random_timestep=config.sample.random_timestep,
            generator=generator,
        )
        
    print("images type:", type(images))
    print("images len:", len(images))
    print("first image shape:", images[0].shape)

    # Convert to PIL
    pil = Image.fromarray(
        (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )

    # Fixed 512x512 for output
    pil = pil.resize((512, 512))

    return pil


# ---------------------------------------------------------
#                   GRADIO UI
# ---------------------------------------------------------
demo = gr.Interface(
    fn=infer,
    inputs=gr.Textbox(lines=2, label="Prompt"),
    outputs=gr.Image(type="pil"),
    title="SD3 + LoRA Inference",
    description="Enter a prompt and generate image using your pipeline_with_logprob",
)

demo.launch()
