from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
# prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

# prompt = "photo of beautiful japanese little girl swimming"
# prompt = "A serene lakeside scene during early morning mist, soft sunlight filtering through trees, gentle ripples on the water, detailed reflections, pastel color palette, cinematic composition, ultra detailed, beautiful lighting, realistic texture, peaceful atmosphere, masterpiece, artstation quality"

prompt = "A beautiful red fox resting in a quiet forest clearing at dawn, soft sunlight filtering through the trees, gentle mist in the air, detailed fur texture, natural color palette, pastel tones, cinematic composition, peaceful atmosphere, ultra detailed, masterpiece, aesthetic lighting, artstation quality"
negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (512, 512),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["1:1"]

images = pipe(
    prompt=prompt + positive_magic["en"],
    # prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    num_images_per_prompt=4,
    # generator=torch.Generator(device="cuda").manual_seed(0)
).images

# image.save("example.png")
for i, img in enumerate(images):
    img.save(f"img{i}.png")


import pdb; pdb.set_trace()