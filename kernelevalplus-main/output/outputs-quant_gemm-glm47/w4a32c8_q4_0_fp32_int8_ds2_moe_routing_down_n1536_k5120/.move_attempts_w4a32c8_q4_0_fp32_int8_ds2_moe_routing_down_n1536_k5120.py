import os
import sys
import shutil
from pathlib import Path

task_name = "w4a32c8_q4_0_fp32_int8_ds2_moe_routing_down_n1536_k5120"
attempts_source = Path("/home/qinhaiyan/kernelevalplus/quant-gemm-attempts")
attempts_target = Path("/home/qinhaiyan/kernelevalplus/output/outputs-quant_gemm-glm47/w4a32c8_q4_0_fp32_int8_ds2_moe_routing_down_n1536_k5120/attempts")

# 无论源目录是否存在，都先创建目标 attempts 目录（保证任务目录下总有 attempts 文件夹）
attempts_target.mkdir(parents=True, exist_ok=True)
print(f"创建/确认目标目录: {attempts_target}")

if not attempts_source.exists():
    print(f"Attempts 源目录不存在: {attempts_source}，跳过移动（目标目录已创建）")
    sys.exit(0)

# 规范化名称再匹配，避免 task 用 hyphen（如 llama3-8b）而 attempts 用 underscore（llama3_8b）导致对不上
def _norm(s):
    return s.replace("-", "_")
task_norm = _norm(task_name)
moved_count = 0
for item in attempts_source.iterdir():
    if not item.is_dir():
        continue
    item_name = item.name
    item_norm = _norm(item_name)
    if item_norm.startswith(task_norm) or task_norm in item_norm:
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
