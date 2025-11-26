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



def dino_cotrain_sd3_patch_fast():
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
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.reference_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_reference_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"

    # config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_geneval.json"
    # config.reference_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_geneval_multinode2"
    # config.test_reference_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_ocr_test"
    
    config.case_name = "fast_dino_cotrain_16_8_lr_times_10_1e4_patch_image_loss_73_again"
    config.save_dir = 'logs/dino/sd3.5-M-fast_dino_cotrain_16_8_lr_times_10_1e4_patch_image_loss_73_again'
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

    config.train.lora_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/logs/dino/sd3.5-M-fast_dino_cotrain_16_8_lr_times_10_1e4_patch_image_loss_73_again/checkpoints/checkpoint-858/lora"
    config.save_folder = "/mnt/bn/vgfm2/test_dit/weijia/outputs_flowgrpo_test2/sd3_dino_pickscore_test_1"
    config.train_d = True
    config.weight_path = None
    config.json_path = "/mnt/bn/vgfm2/test_dit/weijia/flow_grpo/prompt2img_merged_pickscore.json"
    config.reference_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    config.test_reference_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_test"
    
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
    config.reference_image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
    
    config.case_name = "fast_pickscore_cotrain_lr_5e6_last1_16_8"
    config.save_dir = 'logs/pickscore/sd3.5-M-fast_pickscore_cotrain_lr_5e6_last1_16_8'
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




def get_config(name):
    return globals()[name]()

