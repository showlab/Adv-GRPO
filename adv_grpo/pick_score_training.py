import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.utils.data import DataLoader

# ====== 使用你找到的 CLIPCriterion ======
from dataclasses import dataclass
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



def evaluate_pickscore(model, processor, json_file, qwen_dir, sd3_dir, device, max_eval=100):
    """
    简单评估：取前 max_eval 对 Qwen vs SD3 pair，算平均分
    """
    model.eval()
    if hasattr(model, "module"):  # DDP 情况
        model = model.module

    with open(json_file, "r") as f:
        prompt2img = json.load(f)

    prompts = list(prompt2img.keys())[:max_eval]

    qwen_scores, sd3_scores = [], []

    for prompt in prompts:
        filename = prompt2img[prompt]
        qwen_img_path = os.path.join(qwen_dir, filename)
        sd3_img_path = os.path.join(sd3_dir, filename)

        if not (os.path.exists(qwen_img_path) and os.path.exists(sd3_img_path)):
            continue

        qwen_img = Image.open(qwen_img_path).convert("RGB")
        sd3_img = Image.open(sd3_img_path).convert("RGB")

        # 文本 & 图像输入
        text_inputs = processor.tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77
        ).to(device)
        qwen_inputs = processor(images=qwen_img, return_tensors="pt").to(device)
        sd3_inputs = processor(images=sd3_img, return_tensors="pt").to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            qwen_features = model.get_image_features(**qwen_inputs)
            sd3_features = model.get_image_features(**sd3_inputs)

            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            qwen_features = qwen_features / qwen_features.norm(dim=-1, keepdim=True)
            sd3_features = sd3_features / sd3_features.norm(dim=-1, keepdim=True)

            # 相似度分数
            logit_scale = model.logit_scale.exp()
            qwen_score = (logit_scale * (text_features @ qwen_features.T)).item()
            sd3_score = (logit_scale * (text_features @ sd3_features.T)).item()

            qwen_scores.append(qwen_score)
            sd3_scores.append(sd3_score)

    model.train()
    if len(qwen_scores) > 0:
        print(f"[Eval] Qwen avg={sum(qwen_scores)/len(qwen_scores):.4f} "
              f"| SD3 avg={sum(sd3_scores)/len(sd3_scores):.4f}")


@dataclass
class CLIPCriterionConfig:
    _target_: str = "trainer.criterions.clip_criterion.CLIPCriterion"
    is_distributed: bool = False  # 本地先关掉
    label_0_column_name: str = "label_0"
    label_1_column_name: str = "label_1"
    input_ids_column_name: str = "input_ids"
    pixels_0_column_name: str = "pixels_0"
    pixels_1_column_name: str = "pixels_1"
    num_examples_per_prompt_column_name: str = "num_examples_per_prompt"
    in_batch_negatives: bool = False


class CLIPCriterion(_Loss):
    def __init__(self, cfg: CLIPCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_features(model, input_ids, pixels_0_values, pixels_1_values):
        # import pdb; pdb.set_trace()
        # if hasattr(model, "module"):
        #     model = model.module
        all_pixel_values = torch.cat([pixels_0_values, pixels_1_values], dim=0)
        # text_features, all_image_features = model(text_inputs=input_ids, image_inputs=all_pixel_values)
        text_features = model.get_text_features(input_ids=input_ids)
        all_image_features = model.get_image_features(pixel_values=all_pixel_values)
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_features, image_1_features = all_image_features.chunk(2, dim=0)
        return image_0_features, image_1_features, text_features

    @staticmethod
    def gather_features(features):
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features

    # def safe_sync(self, msg):
    #     torch.cuda.synchronize()
    #     print(f"[Rank {dist.get_rank()}] OK at {msg}")

    def calc_loss(
            self,
            text_features,
            image_0_features,
            image_1_features,
            logit_scale,
            label_0,
            label_1,
            num_examples_per_prompt,
            *args,
            **kwargs
    ):
        # self.safe_sync("start")

        device = image_0_features.device

        # gather features
        if self.cfg.is_distributed:
            image_0_features = self.gather_features(image_0_features)
            image_1_features = self.gather_features(image_1_features)
            text_features = self.gather_features(text_features)
            label_0 = self.gather_features(label_0)
            label_1 = self.gather_features(label_1)
            num_examples_per_prompt = self.gather_features(num_examples_per_prompt)

        # calc logits # TODO use local loss as open-clip does
        all_image_features = torch.cat([image_0_features, image_1_features], dim=0)  # (2 * batch_size, dim)
        logits_per_image = logit_scale * all_image_features @ text_features.T
        image_0_logits, image_1_logits = logits_per_image.chunk(2, dim=0)
        text_logits = logit_scale * text_features @ all_image_features.T

        if self.cfg.in_batch_negatives:
            # get labels
            num_images = all_image_features.shape[0]
            image_labels = torch.arange(num_images, device=device, dtype=torch.long)
            image_0_labels, image_1_labels = image_labels.chunk(2, dim=0)
            num_texts = text_features.shape[0]
            text_labels = torch.arange(num_texts, device=device, dtype=torch.long)

            # image loss - we want to increase the logits of the preferred image to the text
            image_0_loss = torch.nn.functional.cross_entropy(image_0_logits, text_labels, reduction="none")
            image_1_loss = torch.nn.functional.cross_entropy(image_1_logits, text_labels, reduction="none")
            # if we have a tie, we will increase both images equally, and average so the image loss of each example is
            # proportional
            image_loss = label_0 * image_0_loss + label_1 * image_1_loss

            # text loss - we want to increase the logits of the text to the preferred image
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, image_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, image_1_labels, reduction="none")

        else:
            text_0_logits, text_1_logits = text_logits.chunk(2, dim=-1)
            index = torch.arange(text_0_logits.shape[0], device=device, dtype=torch.long)

            text_0_logits = text_0_logits[index, index]
            text_1_logits = text_1_logits[index, index]
            text_logits = torch.stack([text_0_logits, text_1_logits], dim=-1)
            text_0_labels = torch.zeros(text_logits.shape[0], device=device, dtype=torch.long)
            text_1_labels = text_0_labels + 1
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, text_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, text_1_labels, reduction="none")

        # if we have a tie we want the logits of for each image to be equal
        text_loss = label_0 * text_0_loss + label_1 * text_1_loss
        # we want the ideal loss to be 0, currently, if there is a tie, it is 0.5 * log(0.5) + 0.5 * log(0.5)
        # so we add log(0.5) to the loss
        is_tie = (label_0 == label_1).float()
        is_tie *= torch.log(torch.tensor(0.5, device=device))
        text_loss += is_tie

        # we average the image and text loss
        if self.cfg.in_batch_negatives:
            loss = (image_loss + text_loss) / 2
        else:
            loss = text_loss
        # import pdb; pdb.set_trace()

        # some prompts have lots of interactions, we want weight them accordingly
        # absolute_example_weight = 1 / num_examples_per_prompt
        # denominator = absolute_example_weight.sum()
        # weight_per_example = absolute_example_weight / denominator
        # loss *= weight_per_example
        loss = loss.mean()
        # import pdb; pdb.set_trace()

        # loss = loss.sum()
        return loss

    def forward(self, model, batch):
        # import pdb; pdb.set_trace()
        image_0_features, image_1_features, text_features = self.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name]
        )
        # print("text_features:", text_features.shape)
        
        loss = self.calc_loss(
            text_features,
            image_0_features,
            image_1_features,
            model.logit_scale.exp(),
            batch[self.cfg.label_0_column_name],
            batch[self.cfg.label_1_column_name],
            batch[self.cfg.num_examples_per_prompt_column_name],
        )
        return loss
    

# ====== 数据准备 ======
class QwenSD3JsonDataset(Dataset):
    def __init__(self, processor, json_file, qwen_dir, sd3_dir):
        """
        json_file: prompt2img.json {prompt: filename}
        qwen_dir: 存放Qwen图像的文件夹
        sd3_dir: 存放SD3图像的文件夹
        """
        self.processor = processor

        with open(json_file, "r") as f:
            self.prompt2img = json.load(f)

        self.prompts = list(self.prompt2img.keys())
        self.qwen_dir = qwen_dir
        self.sd3_dir = sd3_dir

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        filename = self.prompt2img[prompt]

        qwen_img_path = os.path.join(self.qwen_dir, filename)
        sd3_img_path = os.path.join(self.sd3_dir, filename)

        if os.path.exists(qwen_img_path) and os.path.exists(sd3_img_path):
            qwen_img = Image.open(qwen_img_path).convert("RGB")
            sd3_img = Image.open(sd3_img_path).convert("RGB")
        else:
            qwen_img = Image.open(sd3_img_path).convert("RGB")
            sd3_img = Image.open(sd3_img_path).convert("RGB")

        # 文本token
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)

        # 图像预处理
        pixels_0 = self.processor(images=qwen_img, return_tensors="pt")["pixel_values"].squeeze(0)
        pixels_1 = self.processor(images=sd3_img, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "input_ids": input_ids,
            "pixels_0": pixels_0,  # 正样本 (Qwen)
            "pixels_1": pixels_1,  # 负样本 (SD3)
            "label_0": torch.tensor(1.0),  
            "label_1": torch.tensor(0.0),
            "num_examples_per_prompt": torch.tensor(1.0)
        }


# ====== 训练 loop ======
# def finetune_pickscore(json_file, qwen_dir, sd3_dir, epochs=2, batch_size=4, lr=1e-6, device="cuda"):
#     processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
#     model = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device)

#     dataset = QwenSD3JsonDataset(processor,json_file, qwen_dir, sd3_dir)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     criterion = CLIPCriterion(CLIPCriterionConfig())
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#     # import pdb; pdb.set_trace()

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for batch in dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             loss = criterion(model, batch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         print(f"Epoch {epoch} | Loss {total_loss/len(dataloader):.4f}")

#     model.save_pretrained("pickscore_qwen_finetuned")
#     return model

def finetune_pickscore_distributed(json_file, qwen_dir, sd3_dir, epochs=2, batch_size=4, lr=1e-6):
    # 1. 初始化分布式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 2. 准备数据
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    dataset = QwenSD3JsonDataset(processor, json_file, qwen_dir, sd3_dir)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 3. 模型 + DDP
    model = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    criterion = CLIPCriterion(CLIPCriterionConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 4. 训练
    model.train()
    if dist.get_rank() == 0:
        evaluate_pickscore(model, processor, json_file, qwen_dir, sd3_dir, device)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 保证每个 epoch shuffle 一样
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = criterion(model.module, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积loss（先local）
            total_loss += loss.item()

            # 每隔一定步打印一次（rank=0）
            if step % 50 == 0:  # 你可以改成10、100
                # all_reduce 把所有 GPU 的 loss 平均
                avg_loss = torch.tensor(loss.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                if dist.get_rank() == 0:
                    print(f"[Epoch {epoch} | Step {step}/{len(dataloader)}] "
                        f"local_loss={loss.item():.4f} | avg_loss={avg_loss.item():.4f}")

        # 每个 epoch 打印 epoch 平均 loss
        epoch_loss = torch.tensor(total_loss / len(dataloader), device=device)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
        if dist.get_rank() == 0:
            print(f"===> Epoch {epoch} done | avg_epoch_loss={epoch_loss.item():.4f}")
            evaluate_pickscore(model, processor, json_file, qwen_dir, sd3_dir, device)

    # 5. 保存模型（只在 rank=0）
    if dist.get_rank() == 0:
        model.module.save_pretrained("pickscore_qwen_finetuned")

    dist.destroy_process_group()


# ====== 用法示例 ======
if __name__ == "__main__":
    finetune_pickscore_distributed(
        json_file="/mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images/prompt2img.json",
        qwen_dir="/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images",
        sd3_dir="/mnt/bn/vgfm2/test_dit/weijia/outputs/sd3_images",
        epochs=2,
        batch_size=4,
        lr=1e-6,
    )
