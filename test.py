

from huggingface_hub import HfApi, CommitOperationAdd
from tqdm import tqdm
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

folder = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
repo_id = "benzweijia/QWen_Image_PickScore"

operations = []

# 遍历文件收集
for root, _, filenames in os.walk(folder):
    for filename in filenames:
        print(filename)
        full_path = os.path.join(root, filename)
        repo_path = os.path.relpath(full_path, folder)

        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=full_path
            )
        )

# 整体进度条
print(f"Total files: {len(operations)}")
pbar = tqdm(total=len(operations), desc="Preparing upload")

# create_commit 不会自动显示进度，所以我们用 hook 模拟
def hook(_):
    pbar.update(1)

for op in operations:
    hook(op)

# 一次性提交（重要！不会触发 rate limit）
api.create_commit(
    repo_id=repo_id,
    repo_type="dataset",
    operations=operations,
    commit_message="Add dataset in batch",
)
pbar.close()

print("Upload finished.")
