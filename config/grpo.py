import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")


    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.case_name = "sd3_kl_0"

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.0
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3_kl_0'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    config.case_name = "geneval_16_8"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.case_name = "pickscore_qwen_sft_kl_0_only_1_ref"
    config.wandb_init = True
    config.mixed_precision = "bf16"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"

    # config.use_lora = False
    # config.train.learning_rate = 1e-5 
    config.limit = None

    config.resolution = 512
    config.sample.train_batch_size =8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.0
    config.train.ref_update_step = 100
    config.train.algorithm = "sft"
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-pickscore_qwen_sft_kl_0_only_1_ref'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    config.eval_reward_fn = {
        "pickscore": 1.0,
        "image_similarity": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def clipscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.case_name = "clipscore_original"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.02
    config.sample.global_std = True
    config.sample.same_latent = True
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/clipscore/sd3.5-M'
    config.reward_fn = {
        "clipscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def dino_sd3_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = None

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0.0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=20
    config.d_lr=5e-6
    config.train.lora_path = None
    config.tune_layer=-2
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"

    
    config.case_name = "fast_dino_16_8_again"
    config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_dino_16_8_again'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "image_similarity":1,
    }
    config.eval_reward_fn = {
        "image_similarity":1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def dino_cotrain_sd3_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0


    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=10
    config.d_lr=1e-4
    config.train.lora_path = None
    config.tune_layer=-2

    # config.use_lora = False
    # config.train.learning_rate = 1e-5
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"
    
    config.case_name = "fast_dino_cotrain_16_8_lr_times_10_1e4_new_loss_24_9_preprocess"
    config.save_dir = 'logs/dino/sd3.5-M-fast_dino_cotrain_16_8_lr_times_10_1e4_new_loss_16_8_preprocess'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "dino_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1,
        "image_similarity": 1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def dino_cotrain_sd3_patch_style_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0


    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=10
    config.d_lr=1e-4
    config.train.lora_path = None
    config.tune_layer=-2

    # config.use_lora = False
    # config.train.learning_rate = 1e-5
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.limit = 2000
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_scientific_8_multinode/prompt2img_node0.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_scientific_8_multinode"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"
    
    config.case_name = "fast_dino_cotrain_16_8_lr_times_10_1e4_patch_image_loss_73_scientific_2k"
    config.save_dir = 'logs/dino/sd3.5-M-fast_dino_cotrain_16_8_lr_times_10_1e4_patch_image_loss_73_scientific_2k'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "dino_patch_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1,
        "image_similarity": 1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def dinov3_cotrain_sd3_patch_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 8
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0


    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=10
    config.d_lr=1e-4
    config.train.lora_path = None
    config.tune_layer=-2

    # config.use_lora = False
    # config.train.learning_rate = 1e-5
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.limit = None
    # config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged.json"
    # config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_8"
    # config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_test"

    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"
    
    config.case_name = "fast_dinov3_cotrain_16_8_lr_times_10_1e4_again"
    config.save_dir = 'logs/dinov3/sd3.5-M-fast_dinov3_cotrain_16_8_lr_times_10_1e4_again'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "dinov3_patch_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1,
        "image_similarity": 1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def dino_cotrain_sd3_patch_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = False

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 8
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0


    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=10
    config.d_lr=1e-4
    config.train.lora_path = None
    config.tune_layer=-2

    # config.use_lora = False
    # config.train.learning_rate = 1e-5
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.limit = None
    # config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged.json"
    # config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_8"
    # config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_test"

    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_geneval.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_geneval_multinode2"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_test"
    
    config.case_name = "fast_dino_cotrain_16_8_lr_times_10_1e4_patch_image_loss_73_again_geneval"
    config.save_dir = 'logs/dino/sd3.5-M-fast_dino_cotrain_16_8_lr_times_10_1e4_patch_image_loss_73_again_geneval'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "dino_patch_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1,
        # "image_similarity": 1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def siglip_cotrain_sd3_patch_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0


    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=10
    config.d_lr=1e-4
    config.train.lora_path = None
    config.tune_layer=-2

    # config.use_lora = False
    # config.train.learning_rate = 1e-5
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.limit = None
    # config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged.json"
    # config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_8"
    # config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_test"

    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"
    
    config.case_name = "fast_siglip_cotrain_16_8_lr_times_10_1e4_dtimes_10_jitter_reward"
    config.save_dir = 'logs/siglip/sd3.5-M-fast_siglip_cotrain_16_8_lr_times_10_1e4_dtimes_10_jitter_reward'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "siglip_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1,
        "siglip_image_similarity": 1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def dino_cotrain_sd3_multi_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = False

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 8
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0


    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0.0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=10
    config.d_lr=1e-4
    config.train.lora_path = None
    config.tune_layer=(11,)
    config.temperature = 2

    # config.use_lora = False
    # config.train.learning_rate = 1e-5
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"
    
    config.case_name = "fast_dino_cotrain_16_8_lr_times_10_1e4_multi_image_loss_11_only_patch3_tem_2"
    config.save_dir = 'logs/dino/sd3.5-M-fast_dino_cotrain_16_8_lr_times_10_1e4_multi_image_loss_11_only_patch3_tem_2'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "dino_multi_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1,
        "image_similarity": 1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def eval_sd3_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = False

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 8
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    # config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    
    config.sample.test_batch_size = 16
    config.sample.repeat = 1

    config.sample.random_timestep = 0


    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0.0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=10
    config.d_lr=1e-4
    config.train.lora_path = None
    config.tune_layer=-2


    config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/dinov3/sd3.5-M-fast_dinov3_cotrain_16_8_lr_times_10_1e4_again/checkpoints/checkpoint-858/lora"
    config.save_folder = "/mnt/bn/vgfm2/test_dit/weijia/outputs_flowgrpo_test2/sd3_dinov3_pickscore_test_1"
    config.train_d = True
    config.weight_path = None
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"
    
    config.reward_fn = {
        "dino_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1,
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config





def pickscore_cotrain_sd3_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0.0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "pickscore"
    config.d_times=20
    config.d_lr=5e-6
    config.train.lora_path = None
    config.tune_layer=-1
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/pickscore_again/sd3.5-M-fast_1node_8_8/checkpoints/checkpoint-1800/lora"

    config.train_d = True
    config.weight_path = None
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    
    config.case_name = "fast_pickscore_cotrain_lr_5e6_last1_16_8"
    config.save_dir = 'logs/pickscore_again/sd3.5-M-fast_pickscore_cotrain_lr_5e6_last1_16_8'
    # config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_16_8'
    config.reward_fn = {
        "pickscore_cotrain":1,
    }
    config.eval_reward_fn = {
        "pickscore":1
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def discriminator_sd3_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    
    config.mixed_precision = "bf16"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 8
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 0

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0.0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.discriminator = "stylegan"
    config.d_times=15
    config.d_lr=3e-4


    config.train_d = True
    config.weight_path = None
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    
    config.case_name = "fast_pickscore_discriminator_cotrain_stylegan_times_30"
    config.save_dir = 'logs/discriminator_again/sd3.5-M-fast_pickscore_discriminator_cotrain_stylegan_times_30'
    config.reward_fn = {
        "discriminator":1,
    }
    config.eval_reward_fn = {
        "discriminator":0.5,
        "pickscore":0.5
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    
    config.mixed_precision = "bf16"
    config.case_name = "fast_1node_16_8_multireward_11"
    config.wandb_init = True

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.mini_num_image_per_prompt = 8
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = None

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0.0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore_again/sd3.5-M-fast_1node_16_8_multireward_11_ocr_pickscore'
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images"
    config.reward_fn = {
        "pickscore": 0.5,
        "ocr": 0.5,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_multi_reward_fast():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    
    config.mixed_precision = "bf16"
    config.case_name = "fast_1node_num_24_mini_9_ds2_t_10_multireward_514"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 9
    # config.sample.mini_num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    # config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M-fast_1node_num_24_mini_9_ds2_t_10_multireward_514'
    config.reward_fn = {
        "pickscore": 0.5,
        "ocr": 0.1,
        "aesthetic": 0.4
    }
    # config.reward_fn = {
    #     "ocr": 1,
    # }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config




def pickscore_sd3_flux():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.mixed_precision = "bf16"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.case_name = "24_9_t_10_noise_timestep_ratio_0_8_sd3_onlyqwen"
    config.wandb_init = True

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 9
    # config.sample.mini_num_image_per_prompt = 4
    # config.sample.num_batches_per_epoch = 2
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.random_timestep = 2

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1 
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.sample.flux_noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.noise_timestep_ratio = 8
    config.save_dir = 'logs/pickscore/sd3.5-M-24_9_t_10_noise_timestep_ratio_0_8_sd3_onlyqwen'
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images"
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def ocr_sd3_qwen():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.mixed_precision = "bf16"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.case_name = "fast_ocr_only_qwen_only_similarity"
    config.wandb_init = True

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 9
    # config.sample.mini_num_image_per_prompt = 4
    # config.sample.num_batches_per_epoch = 2
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1 
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0.0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.sample.flux_noise_level = 0.8
    # config.sample.random_timestep = 7
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.noise_timestep_ratio = 6
    config.save_dir = 'logs/ocr/sd3.5-M-fast_ocr_only_qwen_only_similarity'
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr"
    config.reward_fn = {
        "ocr": 0.5,
        "image_similarity": 0.5
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def general_ocr_sd3_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_4gpu():
    gpu_number=4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config





def general_ocr_sd3_fast():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.mixed_precision = "bf16"

    config.sample.train_num_steps = 2
    config.case_name = "ocr_baseline_fast_timestep_10_random_4"

    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 9
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.0
    # config.train.beta = 0.0
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.random_timestep = 4

    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/ocr_baseline_fast_timestep_10_random_4'
    config.reward_fn = {
        "ocr": 0.5,
    }
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config




def general_geneval_sd3_fast():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    # config.sample.eval_guidance_scale = 1
    config.mixed_precision = "bf16"
    config.wandb_init=True

    config.sample.train_num_steps = 2
    config.case_name = "geneval_baseline_fast_timestep_10_ref3"

    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.mini_num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.0
    # config.train.beta = 0.0f
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.random_timestep = None

    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/geneval/geneval_baseline_fast_timestep_10_ref3'
    # config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/geneval/geneval_baseline_fast_timestep_10_again/checkpoints/checkpoint-120/lora"
    config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/geneval/geneval_baseline_fast_timestep_10_ref2/checkpoints/checkpoint-120/lora"
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flowgrpo_new/flow_grpo/prompt2img_merged_geneval.json"
    config.external_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_geneval_multinode2"


    config.reward_fn = {
        "geneval": 0.5,
        "image_similarity": 0.5,
    }
    config.eval_reward_fn = {
        "geneval": 1,
    }
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_1gpu():
    gpu_number = 1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.train_num_steps = 1
    config.case_name = "ocr_test"

    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 8
    config.sample.mini_num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_flux():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-group24'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def get_config(name):
    return globals()[name]()
