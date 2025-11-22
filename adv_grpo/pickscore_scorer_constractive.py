from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class PickScoreScorerConstractive(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)
        
    @torch.no_grad()
    def __call__(self, prompt, images, ref_images):
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        ref_image_inputs = self.processor(
            images=ref_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        ref_image_inputs = {k: v.to(device=self.device) for k, v in ref_image_inputs.items()}



        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        ref_image_embs = self.model.get_image_features(**ref_image_inputs)
        ref_image_embs = ref_image_embs / ref_image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # norm to 0-1
        scores = scores/26

        ref_scores = logit_scale * (text_embs @ ref_image_embs.T)
        ref_scores = ref_scores.diag()
        ref_scores = ref_scores/26


        return scores, ref_scores, image_embs, ref_image_embs

# Usage example
def main():
    scorer = PickScoreScorer(
        device="cuda",
        dtype=torch.float32
    )
    images=[
    "nasa.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    print(scorer(prompts, pil_images))

if __name__ == "__main__":
    main()