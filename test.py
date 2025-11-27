

# from huggingface_hub import HfApi, CommitOperationAdd
# from tqdm import tqdm
# import os

# api = HfApi(token=os.getenv("HF_TOKEN"))

# folder = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode"
# repo_id = "benzweijia/QWen_Image_PickScore"

# operations = []

# def fast_scan_folder(folder):
#     """使用 os.scandir() 加速 文件夹递归遍历"""
#     stack = [folder]
#     while stack:
#         current = stack.pop()
#         with os.scandir(current) as it:
#             for entry in it:
#                 if entry.is_dir():
#                     stack.append(entry.path)
#                 else:
#                     yield entry.path  # 返回完整路

# total_files = sum(1 for _ in fast_scan_folder(folder))
# print(f"Total files: {total_files}")
# pbar = tqdm(total=total_files, desc="Scanning & building operations")

# # 遍历文件收集
# for full_path in fast_scan_folder(folder):
#     filename = os.path.basename(full_path)
#     # print(filename)
#     repo_path = os.path.relpath(full_path, folder)

#     operations.append(
#         CommitOperationAdd(
#             path_in_repo=repo_path,
#             path_or_fileobj=full_path
#         )
#     )
#     pbar.update(1)
# pbar.close()

# # 整体进度条
# print(f"Total files: {len(operations)}")
# pbar = tqdm(total=len(operations), desc="Preparing upload")

# # create_commit 不会自动显示进度，所以我们用 hook 模拟
# def hook(_):
#     pbar.update(1)

# for op in operations:
#     hook(op)

# # 一次性提交（重要！不会触发 rate limit）
# api.create_commit(
#     repo_id=repo_id,
#     repo_type="dataset",
#     operations=operations,
#     commit_message="Add dataset in batch",
# )
# pbar.close()

# print("Upload finished.")


from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "benzweijia/QWen_Image_PickScore"
local_tar = "/mnt/bn/vgfm2/test_dit/weijia/outputs/qwen_images_pickscore_8_multinode/images.tar"

api.upload_file(
    path_or_fileobj=local_tar,
    path_in_repo="images.tar",
    repo_id=repo_id,
    repo_type="dataset",
)