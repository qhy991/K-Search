import os
import sys
import shutil
from pathlib import Path

task_name = "w4a32c8_q4_0_fp32_int8_ds3_moe_routing_up_n512_k7168"
model_name = "glm47"
search_dirs = ['/home/qinhaiyan/kernelevalplus/quant-gemm-attempts-glm47', '/home/qinhaiyan/kernelevalplus/quant-gemm-attempts', '/home/qinhaiyan/kernelevalplus/attempts']
attempts_target = Path("/home/qinhaiyan/kernelevalplus/output/outputs-quant_gemm-glm47/w4a32c8_q4_0_fp32_int8_ds3_moe_routing_up_n512_k7168/attempts")

# 无论源目录是否存在，都先创建目标 attempts 目录
attempts_target.mkdir(parents=True, exist_ok=True)
print(f"创建/确认目标目录: {attempts_target}")
print(f"模型: {model_name}")
print(f"搜索目录: {search_dirs}")

# 规范化名称再匹配
def _norm(s):
    return s.replace("-", "_")
task_norm = _norm(task_name)
moved_count = 0

# 按优先级搜索并移动
for attempts_source in search_dirs:
    attempts_path = Path(attempts_source)
    if not attempts_path.exists():
        print(f"源目录不存在，跳过: {attempts_path}")
        continue

    print(f"\n搜索源目录: {attempts_path}")
    for item in attempts_path.iterdir():
        if not item.is_dir():
            continue
        item_name = item.name
        item_norm = _norm(item_name)
        if item_norm.startswith(task_norm) or task_norm in item_norm:
            target_item = attempts_target / item_name
            if target_item.exists():
                print(f"  目标已存在，跳过: {target_item}")
                continue
            try:
                shutil.move(str(item), str(target_item))
                print(f"  已移动: {item} -> {target_item}")
                moved_count += 1
            except Exception as e:
                print(f"  移动失败 {item}: {e}")

print(f"\n共移动 {moved_count} 个文件夹到 {attempts_target}")
