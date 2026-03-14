#!/usr/bin/env python3
"""
将实验根目录下的 attempts/ 中与各 task 对应的子目录，移动到 outputs 下对应任务目录的 attempts/ 中。

用法:
  python3 move_attempts_into_task_folders.py <exp_root> [--op-type rms_norm|topk]
  python3 move_attempts_into_task_folders.py /path/to/minimax-m2.5-rms_norm-20260309
  python3 move_attempts_into_task_folders.py /path/to/glm-5-topk-20260309 --op-type topk

支持结构:
  - rms_norm: outputs/rms_norm/<task>/ 或 outputs/<task>/
  - topk: outputs/topk/<task>/
  - attempts/<task_name>_v1, ...  -> 移动到 <outputs_base>/<task_name>/attempts/
"""

import argparse
import shutil
import sys
from pathlib import Path


def find_output_base(exp_root: Path, op_type: str) -> Path | None:
    """找到任务目录的根。op_type: rms_norm | topk"""
    if not exp_root.is_dir():
        return None
    outputs = exp_root / "outputs"
    if not outputs.is_dir():
        return None
    if op_type == "topk":
        topk = outputs / "topk"
        if topk.is_dir():
            return topk
        for sub in outputs.iterdir():
            if sub.is_dir() and (sub / "topk").is_dir():
                return sub / "topk"
        return None
    # rms_norm
    for sub in exp_root.iterdir():
        if sub.is_dir() and sub.name.startswith("fp32_rms_norm_"):
            return exp_root
    for sub in outputs.iterdir():
        if sub.is_dir() and sub.name.startswith("fp32_rms_norm_"):
            return outputs
    rn = outputs / "rms_norm"
    if rn.is_dir():
        return rn
    for sub in outputs.iterdir():
        if sub.is_dir() and (sub / "rms_norm").is_dir():
            return sub / "rms_norm"
    return None


def get_task_names(output_base: Path, op_type: str) -> list[str]:
    """获取所有任务名。op_type 决定前缀：fp32_rms_norm_ 或 fp32_top_k_sampling_"""
    prefix = "fp32_top_k_sampling_" if op_type == "topk" else "fp32_rms_norm_"
    tasks = []
    for sub in output_base.iterdir():
        if sub.is_dir() and sub.name.startswith(prefix) and not sub.name.endswith(".xlsx"):
            tasks.append(sub.name)
    return sorted(tasks, key=len, reverse=True)  # 长前缀优先匹配


def main():
    parser = argparse.ArgumentParser(description="将 attempts 按任务移动到 outputs 下各 task/attempts/")
    parser.add_argument("exp_root", type=Path, help="实验根目录")
    parser.add_argument("--op-type", choices=("rms_norm", "topk"), default="rms_norm",
                        help="算子类型，用于定位 outputs 与匹配 attempt 名前缀 (默认: rms_norm)")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的操作，不实际移动")
    args = parser.parse_args()

    exp_root = args.exp_root.resolve()
    op_type = args.op_type
    if not exp_root.is_dir():
        print(f"错误: 目录不存在 {exp_root}", file=sys.stderr)
        return 1

    attempts_dir = exp_root / "attempts"
    if not attempts_dir.is_dir():
        print(f"未找到 attempts 目录: {attempts_dir}")
        return 0

    output_base = find_output_base(exp_root, op_type)
    if not output_base:
        print(f"未找到 {op_type} 输出根: {exp_root}")
        return 1

    prefix = "fp32_top_k_sampling_" if op_type == "topk" else "fp32_rms_norm_"
    task_names = get_task_names(output_base, op_type)
    if not task_names:
        print(f"未在 {output_base} 下找到 {prefix}* 任务目录")
        return 1

    # topk 时：attempt 名可能用 qwen2_5_7b / llama3_8b，任务目录可能用 qwen2.5-7b / llama3-8b；还有拼写 fp32_topk_ 需归一化
    def norm_for_match(s: str, op: str) -> str:
        if op != "topk":
            return s
        s = s.replace("fp32_topk_", "fp32_top_k_sampling_")
        s = s.replace("qwen2_5_7b", "qwen2.5-7b").replace("llama3_8b", "llama3-8b")
        return s

    moved = 0
    skipped = 0
    # topk 时也接受拼写 fp32_topk_ 的 attempt 名
    def is_relevant_attempt(n: str) -> bool:
        if n.startswith(prefix):
            return True
        if op_type == "topk" and n.startswith("fp32_topk_"):
            return True
        return False

    for attempt_sub in sorted(attempts_dir.iterdir()):
        if not attempt_sub.is_dir():
            continue
        name = attempt_sub.name
        if not is_relevant_attempt(name):
            continue
        name_norm = norm_for_match(name, op_type)
        # 匹配到哪个任务（最长前缀，用归一化名匹配）
        task_name = None
        for t in task_names:
            if name_norm.startswith(t + "_") and len(name_norm) > len(t) + 1:
                task_name = t
                break
        if not task_name:
            continue
        task_dir = output_base / task_name
        if not task_dir.is_dir():
            print(f"跳过: 任务目录不存在 {task_dir}")
            skipped += 1
            continue
        attempts_dest = task_dir / "attempts"
        dest = attempts_dest / name
        if dest.exists():
            print(f"跳过(已存在): {name} -> {dest}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"[dry-run] 移动: {attempt_sub} -> {dest}")
            moved += 1
            continue
        attempts_dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(attempt_sub), str(dest))
        print(f"已移动: {name} -> {dest.relative_to(exp_root)}")
        moved += 1

    print(f"完成: 移动 {moved} 个, 跳过 {skipped} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
