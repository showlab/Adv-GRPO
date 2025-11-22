from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class PickScoreScorer(torch.nn.Module):
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
    def __call__(self, prompt, images):
        # Preprocess images
        if hasattr(self.model, "module"):
            self.model = self.model.module
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
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
        # image_embs = self.model.get_image_features(**image_inputs)
        import pdb; pdb.set_trace()
        image_embs = self.model.vision_model(image_inputs["pixel_values"],output_hidden_states=True)
        image_embs = image_embs.last_hidden_state

        image_embs = self.model.visual_projection(image_embs)  # [B, N, 1024]
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        # scores = logit_scale * (text_embs @ image_embs.T)
        patch_scores = torch.einsum("bd,bnd->bn", text_embs, image_embs)  # [B, N]
        scores = logit_scale * patch_scores.mean(dim=1)  # 取所有 patch 的平均
        # scores = scores.diag()
        # norm to 0-1
        scores = scores/26
        # import pdb; pdb.set_trace()
        return scores

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