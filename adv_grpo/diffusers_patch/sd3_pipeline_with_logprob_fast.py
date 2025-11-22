# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
import random
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob_new as sde_step_with_logprob
from PIL import Image
from torchvision import transforms



@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    mini_num_image_per_prompt: int = 1,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    train_num_steps: int = 1,
    process_index: int = 0,
    sample_num_steps: int = 10,
    random_timestep: Optional[int] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    # import pdb; pdb.set_trace()
    
    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()
    # import pdb; pdb.set_trace()
    # latents = latents.to(prompt_embeds.dtype)

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    # timesteps = timesteps.to(prompt_embeds.dtype)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    random.seed(process_index)
    if random_timestep is None:
        random_timestep = random.randint(0, sample_num_steps//2)


    # 6. Prepare image embeddings
    all_latents = []
    all_log_probs = []
    all_timesteps = []

    if self.do_classifier_free_guidance:
        tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    # 7. Denoising loop
    # import pdb; pdb.set_trace()
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # import pdb; pdb.set_trace()
        for i, t in enumerate(timesteps):
            if i < random_timestep:
                cur_noise_level = 0
            elif i == random_timestep:
                cur_noise_level= noise_level
                # 将latents repeat mini_num_image_per_prompt次
                latents = latents.repeat(mini_num_image_per_prompt, 1, 1, 1)
                prompt_embeds = prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
                if self.do_classifier_free_guidance:
                    tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                all_latents.append(latents)
            elif i > random_timestep and i < random_timestep + train_num_steps:
                cur_noise_level = noise_level
            else:
                cur_noise_level= 0
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            # import pdb; pdb.set_trace()
            # noise_pred = self.transformer( hidden_states=latent_model_input,  timestep=timestep, encoder_hidden_states=tem_prompt_embeds,pooled_projections=tem_pooled_prompt_embeds, joint_attention_kwargs=self.joint_attention_kwargs,return_dict=False, )[0]
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=tem_prompt_embeds,
                pooled_projections=tem_pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            latents_dtype = latents.dtype

            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0), 
                latents.float(),
                noise_level=cur_noise_level,
            )
                
            # if latents.dtype != latents_dtype:
            #     latents = latents.to(latents_dtype)
            
            if i >= random_timestep and i < random_timestep + train_num_steps:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t.repeat(len(latents)))
                # import pdb; pdb.set_trace()
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
            

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    reconstructd_image = self.image_processor.postprocess(image, output_type="pil")
    # reconstructd_image[0].save("0.png")
    # import pdb; pdb.set_trace()
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()
    return image, all_latents, all_log_probs, all_timesteps



@torch.no_grad()
def pipeline_with_logprob_new(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    mini_num_image_per_prompt: int = 1,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    train_num_steps: int = 1,
    process_index: int = 0,
    sample_num_steps: int = 10,
    random_timestep: Optional[int] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    # import pdb; pdb.set_trace()

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )
    # import pdb; pdb.set_trace()

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    # import pdb; pdb.set_trace()
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    # import pdb; pdb.set_trace()
    
    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    # import pdb; pdb.set_trace()
    # latents = latents.to(prompt_embeds.dtype)

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    # timesteps = timesteps.to(prompt_embeds.dtype)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    random.seed(process_index)
    if random_timestep is None:
        random_timestep = random.randint(0, sample_num_steps//2)


    # 6. Prepare image embeddings
    all_latents = []
    all_log_probs = []
    all_timesteps = []
    # import pdb; pdb.set_trace()

    if self.do_classifier_free_guidance:
        tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    # 7. Denoising loop
    # import pdb; pdb.set_trace()
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # import pdb; pdb.set_trace()
        for i, t in enumerate(timesteps):
            if i < random_timestep:
                cur_noise_level = 0
            elif i == random_timestep:
                cur_noise_level= noise_level
                # 将latents repeat mini_num_image_per_prompt次
                latents = latents.repeat(mini_num_image_per_prompt, 1, 1, 1)
                prompt_embeds = prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
                if self.do_classifier_free_guidance:
                    tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                all_latents.append(latents)
            elif i > random_timestep and i < random_timestep + train_num_steps:
                cur_noise_level = noise_level
            else:
                cur_noise_level= 0
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            # import pdb; pdb.set_trace()
            # noise_pred = self.transformer( hidden_states=latent_model_input,  timestep=timestep, encoder_hidden_states=tem_prompt_embeds,pooled_projections=tem_pooled_prompt_embeds, joint_attention_kwargs=self.joint_attention_kwargs,return_dict=False, )[0]
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=tem_prompt_embeds,
                pooled_projections=tem_pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents_dtype = latents.dtype
           
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0), 
                latents.float(),
                noise_level=cur_noise_level,
            )
                
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            
            if i >= random_timestep and i < random_timestep + train_num_steps:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t.repeat(len(latents)))
                # import pdb; pdb.set_trace()
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
            

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()
    return image, all_latents, all_log_probs, all_timesteps




@torch.no_grad()
def pipeline_with_logprob_random(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    mini_num_image_per_prompt: int = 1,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    train_num_steps: int = 1,
    process_index: int = 0,
    sample_num_steps: int = 10,
    random_timestep: Optional[int] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    # import pdb; pdb.set_trace()

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )
    # import pdb; pdb.set_trace()

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    # import pdb; pdb.set_trace()
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    prompt_embeds = prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
    # import pdb; pdb.set_trace()
    
    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        prompt_embeds.shape[0],
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    # import pdb; pdb.set_trace()
    # latents = latents.to(prompt_embeds.dtype)

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    # timesteps = timesteps.to(prompt_embeds.dtype)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    random.seed(process_index)
    if random_timestep is None:
        random_timestep = random.randint(0, sample_num_steps//2)


    # 6. Prepare image embeddings
    all_latents = []
    all_log_probs = []
    all_timesteps = []
    if self.do_classifier_free_guidance:
        tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    if self.do_classifier_free_guidance:
        tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    # 7. Denoising loop
    # import pdb; pdb.set_trace()
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # import pdb; pdb.set_trace()
        for i, t in enumerate(timesteps):
            if i < random_timestep:
                cur_noise_level = 0
            elif i == random_timestep:
                cur_noise_level= noise_level
                # 将latents repeat mini_num_image_per_prompt次
                # latents = latents.repeat(mini_num_image_per_prompt, 1, 1, 1)
                # prompt_embeds = prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
                # pooled_prompt_embeds = pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
                # negative_prompt_embeds = negative_prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
                # negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
                # if self.do_classifier_free_guidance:
                #     tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                #     tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                all_latents.append(latents)
            elif i > random_timestep and i < random_timestep + train_num_steps:
                cur_noise_level = noise_level
            else:
                cur_noise_level= 0
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            # import pdb; pdb.set_trace()
            # noise_pred = self.transformer( hidden_states=latent_model_input,  timestep=timestep, encoder_hidden_states=tem_prompt_embeds,pooled_projections=tem_pooled_prompt_embeds, joint_attention_kwargs=self.joint_attention_kwargs,return_dict=False, )[0]
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=tem_prompt_embeds,
                pooled_projections=tem_pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents_dtype = latents.dtype
           
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0), 
                latents.float(),
                noise_level=cur_noise_level,
            )
                
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            
            if i >= random_timestep and i < random_timestep + train_num_steps:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t.repeat(len(latents)))
                # import pdb; pdb.set_trace()
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
            

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()
    return image, all_latents, all_log_probs, all_timesteps



def move_scheduler_to_device(scheduler, device="cuda"):
    for attr_name in dir(scheduler):
        attr = getattr(scheduler, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(scheduler, attr_name, attr.to(device))
    return scheduler


def image_to_latent(pipe, images: Union[Image.Image, List[Image.Image]], device="cuda"):
    # 统一转 list
    if isinstance(images, Image.Image):
        images = [images]

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # 转 [0,1]
        transforms.Normalize([0.5], [0.5])  # 映射到 [-1,1]
    ])

    # 批量处理
    img_tensors = [preprocess(img) for img in images]  # list of [3,512,512]
    img_tensor = torch.stack(img_tensors, dim=0).to(device, dtype=torch.float32)  # [B,3,512,512]
    # import pdb; pdb.set_trace()

    # 过 VAE 编码
    latent = pipe.vae.encode(img_tensor).latent_dist.sample()
    latent = latent * pipe.vae.config.scaling_factor
    return latent.to(torch.bfloat16)  # [B,4,64,64]  (假设512输入，缩小8倍)


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
        


@torch.no_grad()
def flux_to_sd3_denoise(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    flux_images=None,
    device="cuda",
    output_type: Optional[str] = "pil",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    random_timestep: Optional[int] = None,
    noise_timestep_ratio: float = 0.4,
    clip_skip: Optional[int] = None,
):
    """
    用 Flux 生成的图像 -> 转 latent -> 加噪 -> 用 SD3 多步去噪
    输出与 pipeline_with_logprob 对齐: image, all_latents, all_log_probs, all_timesteps
    """
    # 1. 转 latent
    flux_latent = image_to_latent(self, flux_images, device)
    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip

    # 2. 准备 scheduler
    noise_scheduler = self.scheduler
    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device)

    # target_idx = torch.tensor([int(noise_timestep_ratio * (len(timesteps) - 1))], device=device)
    target_idx =  torch.tensor([noise_timestep_ratio], device=device)
    t = timesteps[target_idx].to(device)

    noise = torch.randn_like(flux_latent)
    sigmas = get_sigmas(noise_scheduler, t, device, n_dim=flux_latent.ndim, dtype=flux_latent.dtype)
    latents = (1.0 - sigmas) * flux_latent + sigmas * noise
    num_channels_latents = self.transformer.config.in_channels
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # latents = self.prepare_latents(
    #     batch_size,
    #     num_channels_latents,
    #     512,
    #     512,
    #     prompt_embeds.dtype,
    #     device,
    #     None,
    #     None,
    # )



    # import pdb; pdb.set_trace()

    # noisy_latent_vis = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    # noisy_latent_vis = noisy_latent_vis.to(dtype=self.vae.dtype)

    # noisy_image = self.vae.decode(noisy_latent_vis, return_dict=False)[0]
    # noisy_image = self.image_processor.postprocess(noisy_image, output_type="pil")[0]

    # 保存到本地
    # noisy_image.save("noisy_image.png")
    # import pdb; pdb.set_trace()

    # 4. Encode prompts (对齐 pipeline_with_logprob 的处理)
    # lora_scale = (
    #     self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    # )
    lora_scale = None
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    # import pdb; pdb.set_trace()
   

    prompt_embeds = prompt_embeds.repeat(latents.shape[0], 1, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(latents.shape[0], 1)
    negative_prompt_embeds = negative_prompt_embeds.repeat(latents.shape[0], 1, 1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(latents.shape[0], 1)


    if self.do_classifier_free_guidance:
        tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        tem_prompt_embeds = prompt_embeds
        tem_pooled_prompt_embeds = pooled_prompt_embeds

    # 5. 从当前 t 开始去噪
    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device)
    start_idx = (timesteps >= t[0]).nonzero()[-1].item()
    timesteps = timesteps[start_idx:]

    all_latents, all_log_probs, all_timesteps = [], [], []
    noise_scheduler = move_scheduler_to_device(noise_scheduler, device)

    for index, t_cur in enumerate(timesteps):
        # import pdb; pdb.set_trace()
        if index==0:
            all_latents.append(latents)
        
        if index<2:
            cur_noise_level = noise_level
        else:
            cur_noise_level = 0.0

        latent_model_input = (
            torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        )
        t_input = t_cur.expand(latent_model_input.shape[0]).to(device)
        
        latents_dtype = latents.dtype
        model_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=t_input,
            encoder_hidden_states=tem_prompt_embeds,
            pooled_projections=tem_pooled_prompt_embeds,
            return_dict=False,
        )[0]

        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # import pdb; pdb.set_trace()
        

        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            noise_scheduler,
            model_pred.float(),
            t_cur.repeat(len(latents)),
            latents.float(),
            noise_level=noise_level,
        )
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

        if index>=0 and index<2:
        # if index<2:
            # print(model_pred)
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_timesteps.append(t_cur.repeat(len(latents)))
            # import pdb; pdb.set_trace()

    # 6. 最终解码
    denoised_latent = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    denoised_latent = denoised_latent.to(dtype=self.vae.dtype)

    image = self.vae.decode(denoised_latent, return_dict=False)[0]
    # reconstructd_image = self.image_processor.postprocess(image, output_type="pil")[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    return image, all_latents, all_log_probs, all_timesteps





@torch.no_grad()
def flux_to_sd3_denoise_random(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    flux_images=None,
    device="cuda",
    output_type: Optional[str] = "pil",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    random_timestep: Optional[int] = None,
    noise_timestep_ratio: float = 0.4,
    clip_skip: Optional[int] = None,
):
    """
    用 Flux 生成的图像 -> 转 latent -> 加噪 -> 用 SD3 多步去噪
    输出与 pipeline_with_logprob 对齐: image, all_latents, all_log_probs, all_timesteps
    """
    # 1. 转 latent
    flux_latent = image_to_latent(self, flux_images, device)
    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip

    # 2. 准备 scheduler
    noise_scheduler = self.scheduler
    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device)

    # target_idx = torch.tensor([int(noise_timestep_ratio * (len(timesteps) - 1))], device=device)
    # t = timesteps[target_idx].to(device)

    # noise = torch.randn_like(flux_latent)
    # sigmas = get_sigmas(noise_scheduler, t, device, n_dim=flux_latent.ndim, dtype=flux_latent.dtype)
    # latents = (1.0 - sigmas) * flux_latent + sigmas * noise
    
    target_idx = torch.tensor([random.randint(5, 10)], device=device)
    t = timesteps[target_idx].to(device)
    # 生成标准高斯噪声
    noise = torch.randn_like(flux_latent)
    # 获取对应的 sigma
    sigmas = get_sigmas(noise_scheduler, t, device, n_dim=flux_latent.ndim, dtype=flux_latent.dtype)
    # 给 latent 加噪
    latents = (1.0 - sigmas) * flux_latent + sigmas * noise 

    # import pdb; pdb.set_trace()

    # noisy_latent_vis = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    # noisy_latent_vis = noisy_latent_vis.to(dtype=self.vae.dtype)

    # noisy_image = self.vae.decode(noisy_latent_vis, return_dict=False)[0]
    # noisy_image = self.image_processor.postprocess(noisy_image, output_type="pil")[0]

    # 保存到本地
    # noisy_image.save("noisy_image.png")
    # import pdb; pdb.set_trace()

    # 4. Encode prompts (对齐 pipeline_with_logprob 的处理)
    # lora_scale = (
    #     self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    # )
    lora_scale = None
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    # import pdb; pdb.set_trace()
   

    prompt_embeds = prompt_embeds.repeat(latents.shape[0], 1, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(latents.shape[0], 1)
    negative_prompt_embeds = negative_prompt_embeds.repeat(latents.shape[0], 1, 1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(latents.shape[0], 1)


    if self.do_classifier_free_guidance:
        tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        tem_prompt_embeds = prompt_embeds
        tem_pooled_prompt_embeds = pooled_prompt_embeds

    # 5. 从当前 t 开始去噪
    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device)
    start_idx = (timesteps >= t[0]).nonzero()[-1].item()
    timesteps = timesteps[start_idx:]

    all_latents, all_log_probs, all_timesteps = [], [], []
    noise_scheduler = move_scheduler_to_device(noise_scheduler, device)

    for index, t_cur in enumerate(timesteps):
        if index==0:
            all_latents.append(latents)
        
        latent_model_input = (
            torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        )
        t_input = t_cur.expand(latent_model_input.shape[0]).to(device)
        
        latents_dtype = latents.dtype
        model_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=t_input,
            encoder_hidden_states=tem_prompt_embeds,
            pooled_projections=tem_pooled_prompt_embeds,
            return_dict=False,
        )[0]

        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # import pdb; pdb.set_trace()
        

        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            noise_scheduler,
            model_pred.float(),
            t_cur.repeat(len(latents)),
            latents.float(),
            noise_level=noise_level,
        )
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

        # if index>=2 and index<4:
        if index<2:
            # print(model_pred)
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_timesteps.append(t_cur.repeat(len(latents)))
            # import pdb; pdb.set_trace()

    # 6. 最终解码
    denoised_latent = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    denoised_latent = denoised_latent.to(dtype=self.vae.dtype)

    image = self.vae.decode(denoised_latent, return_dict=False)[0]
    # reconstructd_image = self.image_processor.postprocess(image, output_type="pil")[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    return image, all_latents, all_log_probs, all_timesteps
