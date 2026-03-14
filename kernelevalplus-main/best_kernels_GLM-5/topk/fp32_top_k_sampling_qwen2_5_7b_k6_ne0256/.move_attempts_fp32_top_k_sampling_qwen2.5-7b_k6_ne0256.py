import os
import sys
import shutil
from pathlib import Path

task_name = "fp32_top_k_sampling_qwen2.5-7b_k6_ne0256"
attempts_source = Path("/home/qinhaiyan/kernelevalplus/attempts")
attempts_target = Path("output/glm-5/topk/fp32_top_k_sampling_qwen2.5-7b_k6_ne0256/attempts")

if not attempts_source.exists():
    print(f"Attempts 目录不存在: {attempts_source}")
    sys.exit(0)

# 创建目标目录
attempts_target.mkdir(parents=True, exist_ok=True)
print(f"创建目标目录: {attempts_target}")

# 查找所有与任务相关的文件夹（以任务名开头或包含任务名）
moved_count = 0
for item in attempts_source.iterdir():
    if not item.is_dir():
        continue
    # 检查文件夹名是否与任务相关（以任务名开头，或包含任务名）
    item_name = item.name
    if item_name.startswith(task_name) or task_name in item_name:
        target_item = attempts_target / item_name
        if target_item.exists():
            print(f"目标已存在，跳过: {target_item}")
            continue
        try:
            shutil.move(str(item), str(target_item))
            print(f"已移动: {item} -> {target_item}")
            moved_count += 1
        except Exception as e:
            print(f"移动失败 {item}: {e}")

print(f"共移动 {moved_count} 个文件夹到 {attempts_target}")
