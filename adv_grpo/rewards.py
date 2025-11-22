from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from adv_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score():
    from adv_grpo.clip_scorer import ClipScorer

    # scorer = ClipScorer(dtype=torch.float32).cuda()
    scorer = ClipScorer().cuda()

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def siglip_image_similarity_score(device):
    import torch
    import numpy as np
    from transformers import SiglipModel
    import torch.nn.functional as F

    # 1. 加载 SigLIP 模型（推荐 so400m-p14-384）
    scorer = SiglipModel.from_pretrained(
        "google/siglip-so400m-patch14-384"
    ).to(device).to(torch.bfloat16)
    scorer.eval()

    # SigLIP preprocess mean/std
    siglip_mean = [0.5, 0.5, 0.5]
    siglip_std  = [0.5, 0.5, 0.5]

    # 模型输入分辨率（自动匹配所选 SigLIP 模型）
    image_size = scorer.config.vision_config.image_size   # e.g., 224/256/384

    def _preprocess(images):
        # 转成 tensor
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)

        # 0-255 → 0-1
        if images.max() > 1.0:
            images = images / 255.0

        # NHWC → NCHW
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        # resize to SigLIP input
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False
        )

        # normalize using SigLIP's mean/std
        mean = torch.tensor(siglip_mean, device=device)[None, :, None, None]
        std  = torch.tensor(siglip_std,  device=device)[None, :, None, None]
        images = (images - mean) / std

        return images.to(device).to(torch.bfloat16)

    def _fn(images, ref_images):
        # 2. preprocess
        images = _preprocess(images)
        ref_images = _preprocess(ref_images)

        with torch.no_grad():
            # SigLIP extract feature
            out_img = scorer.vision_model(
                pixel_values=images.to(torch.float32)
            )
            out_ref = scorer.vision_model(
                pixel_values=ref_images.to(torch.float32)
            )

            emb_images = out_img.pooler_output.to(torch.bfloat16)
            emb_ref = out_ref.pooler_output.to(torch.bfloat16)

        # 3. normalize embeddings (L2)
        emb_images = emb_images / emb_images.norm(dim=-1, keepdim=True)
        emb_ref = emb_ref / emb_ref.norm(dim=-1, keepdim=True)

        # 4. cosine similarity
        scores = torch.matmul(emb_images, emb_ref.T)  # [N,M]
        per_img = scores.max(dim=1).values            # [N]

        return per_img.detach(), {"pairwise": scores.detach()}

    return _fn



def image_similarity_score(device):
    import torch
    import numpy as np
    import timm
    import torch.nn.functional as F

    # 1. 加载 DINOv2 模型（这里用 ViT-Base/14，可换成 ViT-L/14 或 ViT-G/14）
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
    # model = timm.create_model('vit_large_patch16_dinov3_qkvb.lvd1689m', pretrained=True)
    model.eval().to(device)


    def _preprocess(images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0
        if images.shape[-1] == 3:  # NHWC -> NCHW
            images = images.permute(0, 3, 1, 2)
        # 调整到 518×518
        # images = F.interpolate(images, size=(518, 518), mode="bicubic", align_corners=False)
        # DINOv2 normalization
        # images = (images - 0.5) / 0.5
        images = torch.nn.functional.interpolate(images, size=(512, 512), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device)[None, :, None, None]
        images = (images - mean) / std
        return images.to(device)
    
    def _fn(images, ref_images):
        # 2. 预处理
        # import pdb; pdb.set_trace()
        images = _preprocess(images)
        ref_images = _preprocess(ref_images)

        with torch.no_grad():
            # import pdb; pdb.set_trace()
            emb_images = model(images)        # [N,D]
            emb_ref = model(ref_images)       # [M,D]

        # 3. 归一化
        emb_images = emb_images / emb_images.norm(dim=-1, keepdim=True)
        emb_ref = emb_ref / emb_ref.norm(dim=-1, keepdim=True)

        # 4. 计算相似度 (余弦相似度)
        scores = torch.matmul(emb_images, emb_ref.T)  # [N,M]
        per_img = scores.max(dim=1).values            # [N]；若想平均用：scores.mean(dim=1)
        # import pdb; pdb.set_trace()

        # 返回一维分数，pairwise 放到 info 里
        # return per_img.detach(), {"pairwise": scores.detach()}, emb_images, emb_ref
        return per_img.detach(), {"pairwise": scores.detach()}


        # return scores, {}

    return _fn




def image_similarity_score_eval(device):
    import torch
    import numpy as np
    import timm
    import torch.nn.functional as F

    # 1. 加载 DINOv2 模型（这里用 ViT-Base/14，可换成 ViT-L/14 或 ViT-G/14）
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
    model.eval().to(device)

    def _preprocess(images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0
        if images.shape[-1] == 3:  # NHWC -> NCHW
            images = images.permute(0, 3, 1, 2)
        # 调整到 518×518
        # images = F.interpolate(images, size=(518, 518), mode="bicubic", align_corners=False)
        # DINOv2 normalization
        # images = (images - 0.5) / 0.5
        images = torch.nn.functional.interpolate(images, size=(518, 518), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device)[None, :, None, None]
        images = (images - mean) / std
        return images.to(device)
    
    def _fn(images, ref_images):
        # 2. 预处理
        # import pdb; pdb.set_trace()
        images = _preprocess(images)
        ref_images = _preprocess(ref_images)

        with torch.no_grad():
            # import pdb; pdb.set_trace()
            emb_images = model(images)        # [N,D]
            emb_ref = model(ref_images)       # [M,D]

        # 3. 归一化
        # emb_images = emb_images / emb_images.norm(dim=-1, keepdim=True)
        # emb_ref = emb_ref / emb_ref.norm(dim=-1, keepdim=True)

        # 4. 计算相似度 (余弦相似度)
        scores = torch.matmul(emb_images, emb_ref.T)  # [N,M]
        per_img = scores.max(dim=1).values            # [N]；若想平均用：scores.mean(dim=1)
        # import pdb; pdb.set_trace()

        # 返回一维分数，pairwise 放到 info 里
        return per_img.detach(), {"pairwise": scores.detach()}, emb_images, emb_ref
        # return per_img.detach(), {"pairwise": scores.detach()}


        # return scores, {}

    return _fn



def dino_cotrain_score(device):
    def _preprocess(images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0
        if images.shape[-1] == 3:  # NHWC -> NCHW
            images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=(518, 518), mode="bicubic", align_corners=False)
        # images = (images - 0.5) / 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device)[None, :, None, None]
        images = (images - mean) / std
        return images.to(device).to(torch.bfloat16)

    def _fn(scorer, head, images, prompts, metadata):
        images = _preprocess(images)

        with torch.no_grad():
            emb = scorer(images)  # [N,D]
        emb = emb / emb.norm(dim=-1, keepdim=True)

        # Head 输出 reward
        scores = head(emb).squeeze(-1)  # [N]
        # import pdb; pdb.set_trace()

        return scores.detach(), {"embeddings": emb.detach()}

    return _fn




def siglip_cotrain_score(device):
    """
    使用方式与原来的 dino_cotrain_score 一致：
    reward_fn = siglip_cotrain_score(device)
    reward_fn(scorer, head, images, prompts, metadata)

    scorer: SigLIPModel (从 HF transformers 加载)
    head: 你的 reward head
    """
    from torchvision import transforms
    tiny_jitter = transforms.ColorJitter(
        brightness=0.02,
        contrast=0.02,
    )

    def _preprocess(images, image_size=224):
        # 转 tensor
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)

        # 将 uint8 0-255 → 0-1
        if images.max() > 1.0:
            images = images / 255.0

        # NHWC → NCHW
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        imgs_aug = []
        for img in images:
            imgs_aug.append(tiny_jitter(img))   # 做轻微亮度扰动
        images = torch.stack(imgs_aug)

        # Resize到 SigLIP 默认输入大小（通常 224 / 256 / 384）
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False
        )

        # ★★★ SigLIP 官方 mean/std（注意不同于 CLIP）★★★
        mean = torch.tensor([0.5, 0.5, 0.5], device=device)[None, :, None, None]
        std  = torch.tensor([0.5, 0.5, 0.5], device=device)[None, :, None, None]

        images = (images - mean) / std

        return images.to(device).to(torch.bfloat16)

    def _fn(scorer, head, images, prompts, metadata):
        """
        scorer: SigLIPModel
        head: reward head
        """

        # 图像预处理（scorer.config.vision_config.image_size 通常 224）
        # import pdb; pdb.set_trace()
        images = _preprocess(images, 512)

        scorer.eval()
        with torch.no_grad():
            # ★★★ SigLIP 特征获取 ★★★
            # 返回 CLS token 的 embedding，类似 CLIP 的 global feature
            vision_out = scorer.vision_model(
                pixel_values=images.to(torch.float32)
            )
            emb = vision_out.pooler_output.to(torch.bfloat16)   # [B, D]

        # head 输出 reward
        scores = head(emb).squeeze(-1)

        return scores.detach(), {"embeddings": emb.detach()}

    return _fn

def dinov3_patch_cotrain_score(device, n_patches=64):
    import torch
    import torch.nn.functional as F

    def _preprocess(images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0
        if images.shape[-1] == 3:  # NHWC -> NCHW
            images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=(512, 512), mode="bicubic", align_corners=False)
        # images = (images - 0.5) / 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device)[None, :, None, None]
        images = (images - mean) / std
        return images.to(device).to(torch.bfloat16)

    def _fn(scorer, head, images, prompts, metadata, cls_weight=0.7):
        images = _preprocess(images)
        with torch.no_grad():
            # 提取所有特征: [B, N+1, D]
            feats = scorer.forward_features(images)

        # --- 分离 CLS 与 patch ---
        cls_emb = feats[:, 0, :]       # [B, D]
        patch_emb = feats[:, 1:, :]    # [B, N, D]
        B, N, D = patch_emb.shape

        # --- 随机采样 patch ---
        n_select = min(n_patches, N)
        idx = torch.randint(0, N, (B, n_select), device=device)
        sampled_patches = torch.gather(
            patch_emb, 1, idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, n_select, D]

        # --- 归一化 ---
        cls_emb = cls_emb / (cls_emb.norm(dim=-1, keepdim=True) + 1e-6)
        sampled_patches = sampled_patches / (sampled_patches.norm(dim=-1, keepdim=True) + 1e-6)

        # --- 计算分数 ---
        cls_score = head(cls_emb).squeeze(-1)                  # [B]
        patch_scores = head(sampled_patches).squeeze(-1)       # [B, n_select]
        patch_score_mean = patch_scores.mean(dim=1)            # [B]

        # --- 混合 reward ---
        hybrid_score = cls_weight * cls_score + (1 - cls_weight) * patch_score_mean
        # hybrid_score = hybrid_score.unsqueeze(-1)               # [B, 1]

        # import pdb; pdb.set_trace()

        # --- 返回结果 ---
        return hybrid_score.detach(), {
            "cls_score": cls_score.detach(),
            "patch_scores": patch_scores.detach(),
            "patch_indices": idx.detach(),
            "cls_weight": cls_weight,
        }

    return _fn


def dino_patch_cotrain_score(device, n_patches=64):
    import torch
    import torch.nn.functional as F

    def _preprocess(images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0
        if images.shape[-1] == 3:  # NHWC -> NCHW
            images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=(518, 518), mode="bicubic", align_corners=False)
        # images = (images - 0.5) / 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device)[None, :, None, None]
        images = (images - mean) / std
        return images.to(device).to(torch.bfloat16)

    def _fn(scorer, head, images, prompts, metadata, cls_weight=0.7):
        images = _preprocess(images)
        with torch.no_grad():
            # 提取所有特征: [B, N+1, D]
            feats = scorer.forward_features(images)

        # --- 分离 CLS 与 patch ---
        cls_emb = feats[:, 0, :]       # [B, D]
        patch_emb = feats[:, 1:, :]    # [B, N, D]
        B, N, D = patch_emb.shape

        # --- 随机采样 patch ---
        n_select = min(n_patches, N)
        idx = torch.randint(0, N, (B, n_select), device=device)
        sampled_patches = torch.gather(
            patch_emb, 1, idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, n_select, D]

        # --- 归一化 ---
        cls_emb = cls_emb / (cls_emb.norm(dim=-1, keepdim=True) + 1e-6)
        sampled_patches = sampled_patches / (sampled_patches.norm(dim=-1, keepdim=True) + 1e-6)

        # --- 计算分数 ---
        cls_score = head(cls_emb).squeeze(-1)                  # [B]
        patch_scores = head(sampled_patches).squeeze(-1)       # [B, n_select]
        patch_score_mean = patch_scores.mean(dim=1)            # [B]

        # --- 混合 reward ---
        hybrid_score = cls_weight * cls_score + (1 - cls_weight) * patch_score_mean
        # hybrid_score = hybrid_score.unsqueeze(-1)               # [B, 1]

        # import pdb; pdb.set_trace()

        # --- 返回结果 ---
        return hybrid_score.detach(), {
            "cls_score": cls_score.detach(),
            "patch_scores": patch_scores.detach(),
            "patch_indices": idx.detach(),
            "cls_weight": cls_weight,
        }

    return _fn


def _get_layer_tokens_timm(model, imgs, layer_ids=(2, 5, 8, 11)):
    handles, feats = [], {i: None for i in layer_ids}

    def make_hook(i):
        def hook(_module, _inp, out):
            # 对 timm 的 ViT，block 的输出通常是 [B, N+1, D]
            feats[i] = out
        return hook

    for i in layer_ids:
        assert 0 <= i < len(model.blocks), f"layer id {i} out of range"
        handles.append(model.blocks[i].register_forward_hook(make_hook(i)))

    # 触发前向。timm 的 ViT 有 forward_features；没有就直接 __call__
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            _ = model.forward_features(imgs)
        else:
            _ = model(imgs)

    for h in handles:
        h.remove()

    return [feats[i] for i in layer_ids]  # list of [B, N+1, D]

# -------- reward 工厂：分层 head + top-k 池化 + 融合 --------
def dino_multi_cotrain_score(
    device,
    topk_tau=0.2,            # 每层取前 tau 比例的 patch logits 做均值
    apply_sigmoid=True,      # 是否把 logit 过 sigmoid 得到 [0,1] reward
    lambda_cls=0.5, 
    zscore=False,            # 是否对 batch 维做 z-score（组内可再自行处理）
):

    def _preprocess(images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0
        if images.shape[-1] == 3:  # NHWC -> NCHW
            images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=(518, 518), mode="bicubic", align_corners=False)
        images = (images - 0.5) / 0.5
        # mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[None, :, None, None]
        # std  = torch.tensor([0.229, 0.224, 0.225], device=images.device)[None, :, None, None]
        # images = (images - mean) / std
        return images.to(device)

    @torch.no_grad()
    def _fn(scorer, heads, fusion, images,prompts=None, metadata=None, layer_ids=(8,),temperature=0.2):
        """
        scorer : timm 的 ViT-DINOv2 backbone（已 .eval() 且 requires_grad=False）
        heads  : nn.ModuleList，长度 = len(layer_ids)，每层一个 head
        fusion : 融合器，将 (B,T) -> (B,)
        images : [N,H,W,3] or [N,3,H,W]，值域 [0,1]/[0,255]
        返回:
          rewards: [N]（float32）
          aux: dict，包含 per_layer_scores 等，便于调试/可视化
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        hmod = heads.module if isinstance(heads, DDP) else heads
        fmod = fusion.module if isinstance(fusion, DDP) else fusion
        x = _preprocess(images).to(dtype=next(scorer.parameters()).dtype, device=device)

        # 取多层 tokens
        tokens_list = _get_layer_tokens_timm(scorer, x, layer_ids=layer_ids)  # list of [B, N+1, D]
        B = x.size(0)
        T = len(tokens_list)

        per_layer_scores = []
        per_layer_logits = []  # 保存每层的 top-k 前的 patch logits（可选）
        per_layer_cls_scores = []

        for t in range(T):
            tokens = tokens_list[t]          # [B, N+1, D]
            patch = tokens[:, 1:]            # [B, N, D]  忽略 CLS
            class_patch  = tokens[:, 0]    
            Bn, N, D = patch.shape

            # head 支持 [B,N,D]：输出 [B,N]
            logits_patch = hmod[t](patch).squeeze(-1)  # [B, N]
            per_layer_logits.append(logits_patch)

            # 层内 top-k 池化
            k = max(1, int(N * topk_tau))
            pooled = logits_patch.topk(k, dim=1).values.mean(dim=1)  # [B]
            per_layer_scores.append(pooled)

            cls_logit = hmod[t](class_patch).squeeze(-1)  # [B]
            per_layer_cls_scores.append(cls_logit)

        per_layer_scores = torch.stack(per_layer_scores, dim=1)  # [B, T]
        per_layer_cls_scores   = torch.stack(per_layer_cls_scores,   dim=1)

        # 融合为最终 logit
        logit_patch = fmod(per_layer_scores)  # [B]
        logit_cls  = fmod(per_layer_cls_scores)

        # logits = (1.0 - float(lambda_cls)) * logit_patch + float(lambda_cls) * logit_cls  # [B]
        logits = logit_patch
        # 标定成 reward
        rewards = logits
        # import pdb; pdb.set_trace()
        if apply_sigmoid:
            rewards = torch.sigmoid(rewards / float(temperature))
        if zscore:
            mu = rewards.mean(dim=0, keepdim=True)
            sigma = rewards.std(dim=0, keepdim=True).clamp_min(1e-6)
            rewards = (rewards - mu) / sigma

        # 输出 float32，避免后续与 bf16 混用出问题
        rewards = rewards.float()
        # import pdb; pdb.set_trace()

        aux = {
            "per_layer_scores": per_layer_scores.float(),  # [B,T]
            "logits": logits.float(),                      # 未标定的融合 logit
            # 下行可能很大（B×T×N），需要时再用；默认不返回节省带宽
            # "per_layer_logits": [lp.float() for lp in per_layer_logits],
        }
        return rewards, aux

    return _fn

def pickscore_score(device):
    from adv_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def pickscore_cotrain_score(device):


    def _fn(scorer, images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        # import pdb; pdb.set_trace()
        return scores, {}

    return _fn


def pickscore_score_patch(device):
    from adv_grpo.pickscore_scorer_patch import PickScoreScorer


    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn




def discriminator_score(device):
    def _fn(scorer, images, prompts=None, metadata=None):
        # 归一化到 [-1,1]
        if isinstance(images, torch.Tensor):
            if images.max() > 1.5:  # 可能是 0~255
                images = images / 255.0
            images = (images - 0.5) * 2.0
        else:
            raise ValueError("images must be a torch.Tensor in [B,3,H,W]")

        with torch.no_grad():
            logits = scorer(images.to(device))  # StyleGAN: [B] / [B,1]  PatchGAN: [B,1,H',W']

            if logits.ndim == 1:  
                # StyleGAN D，已经是 [B]
                scores = torch.sigmoid(logits)
            elif logits.ndim == 2 and logits.shape[1] == 1:  
                # StyleGAN D，输出 [B,1]
                scores = torch.sigmoid(logits.squeeze(1))
            elif logits.ndim == 4 and logits.shape[1] == 1:  
                # PatchGAN D，输出 [B,1,H',W']
                scores = torch.sigmoid(logits).mean(dim=[1,2,3])  # -> [B]
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

        return scores.cpu(), {}

    return _fn



def imagereward_score(device):
    from adv_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from adv_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from adv_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        # import pdb; pdb.set_trace()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from adv_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def constractive_external(device, beta=0.5, top_n=2):
    import torch
    from PIL import Image
    from adv_grpo.pickscore_scorer_constractive import PickScoreScorerConstractive

    scorer = PickScoreScorerConstractive(dtype=torch.float32, device=device)

    def _fn(images,ref_images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]

            ref_images = (ref_images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_images = ref_images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            ref_images = [Image.fromarray(image) for image in ref_images]
        # import pdb; pdb.set_trace()

        # scorer 输出
        scores, ref_scores, img_embeds, ref_img_embeds = scorer(prompts, images, ref_images)

        # external anchor
        ref_embed = ref_img_embeds.mean(dim=0, keepdim=True)  # (1,D)
        ext_score = ref_scores.mean()

        # 找 group 内 Top-N 最高分的候选
        top_idx = torch.topk(scores, k=min(top_n, len(scores))).indices
        hack_scores = scores[top_idx]
        hack_embeds = img_embeds[top_idx]  # (N,D)

        # 如果 external 已经比所有 hack 分数都高 → 不修正
        if ext_score >= hack_scores.max():
            return scores, {"raw_scores": scores, "ref_scores": ref_scores}

        # 计算对比修正
        sim_to_ext = torch.nn.functional.cosine_similarity(img_embeds, ref_embed)
        sim_to_hack = torch.nn.functional.cosine_similarity(
            img_embeds.unsqueeze(1), hack_embeds.unsqueeze(0), dim=-1
        )  # (num_images, N)
        sim_to_hack = sim_to_hack.mean(dim=1)  # 平均 N 个负样本相似度

        adjusted_scores = scores + beta * (sim_to_ext - sim_to_hack)

        return adjusted_scores, {
            "raw_scores": scores,
            "ref_scores": ref_scores,
            "sim_to_ext": sim_to_ext,
            "sim_to_hack": sim_to_hack,
            "hack_scores": hack_scores
        }

    return _fn


def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn



def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
        "image_similarity_eval": image_similarity_score_eval,
        "constractive_external": constractive_external,
        "discriminator": discriminator_score,
        "pickscore_cotrain": pickscore_cotrain_score,
        "pickscore_patch":pickscore_score_patch,
        "dino_cotrain":dino_cotrain_score,
        "dino_multi_cotrain": dino_multi_cotrain_score,
        "dino_patch_cotrain": dino_patch_cotrain_score,
        "dinov3_patch_cotrain": dinov3_patch_cotrain_score,
        "siglip_cotrain": siglip_cotrain_score,
        "siglip_image_similarity": siglip_image_similarity_score
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        # import pdb; pdb.set_trace()
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, scorer = None, ref_images=None, only_strict=True, head=None, fusion=None, layer_ids = None, temperature=0.2):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            elif score_name == "siglip_image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            elif score_name == "image_similarity_eval":
                scores, rewards, feat, ref_feat = score_fns[score_name](images, ref_images)
                score_details['feat'] = feat
                score_details['ref_feat'] = ref_feat
            elif score_name == "constractive_external":
                scores, rewards = score_fns[score_name](images, prompts, ref_images)
            elif score_name == "discriminator":
                scores, rewards = score_fns[score_name](scorer, images, prompts, ref_images)
            elif score_name == "pickscore_cotrain":
                scores, rewards = score_fns[score_name](scorer, images, prompts, metadata)
            elif score_name == "dino_cotrain":
                scores, rewards = score_fns[score_name](scorer, head, images, prompts, metadata)
            elif score_name == "siglip_cotrain":
                scores, rewards = score_fns[score_name](scorer, head, images, prompts, metadata)
            elif score_name == "dino_multi_cotrain":
                scores, rewards = score_fns[score_name](scorer, head, fusion, images, prompts, metadata, layer_ids, temperature)
            elif score_name == "dino_patch_cotrain":
                scores, rewards = score_fns[score_name](scorer, head, images, prompts, metadata)
            elif score_name == "dinov3_patch_cotrain":
                scores, rewards = score_fns[score_name](scorer, head, images, prompts, metadata)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)

            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        # import pdb; pdb.set_trace()
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()