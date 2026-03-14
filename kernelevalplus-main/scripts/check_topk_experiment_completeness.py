#!/usr/bin/env python3
"""
根据 kernelevalplus/definitions/topk 下的定义，检查每个 topk 实验目录是否完整。
完整 = 每个定义对应的任务目录存在且含有 kernel 文件（.cu）。
"""

import json
import sys
from pathlib import Path

DEFINITIONS_DIR = Path(__file__).resolve().parents[1] / "definitions" / "topk"
KERNELEVAL_EXP = Path("/home/qinhaiyan/KERNELEVAL-exp")


def get_definition_task_names() -> list[str]:
    """从 definitions/topk 下所有 JSON 的 name 字段收集任务名（唯一、有序）。"""
    names = []
    seen = set()
    for j in sorted(DEFINITIONS_DIR.rglob("*.json")):
        with open(j, "r", encoding="utf-8") as f:
            d = json.load(f)
        n = d.get("name")
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    return names


def find_topk_output_base(exp_root: Path) -> Path | None:
    """在实验根目录下找到 topk 输出根：outputs/topk 或 outputs/<某层>/topk。"""
    outputs = exp_root / "outputs"
    if not outputs.is_dir():
        return None
    # 直接 outputs/topk
    topk = outputs / "topk"
    if topk.is_dir():
        return topk
    # 或 outputs/<model>/topk
    for sub in outputs.iterdir():
        if sub.is_dir():
            t = sub / "topk"
            if t.is_dir():
                return t
    return None


def task_has_kernel(task_dir: Path) -> bool:
    """任务目录或其子目录中是否存在 .cu 文件。"""
    if not task_dir.is_dir():
        return False
    for p in task_dir.rglob("*.cu"):
        if p.is_file():
            return True
    return False


def check_experiment(exp_root: Path, task_names: list[str]) -> dict:
    """检查一个实验的完整性，返回统计与缺失列表。"""
    topk_base = find_topk_output_base(exp_root)
    if not topk_base:
        return {
            "exp_root": str(exp_root),
            "topk_base": None,
            "total": len(task_names),
            "completed": 0,
            "missing": list(task_names),
            "passed": [],
            "has_outputs": False,
        }
    completed = []
    missing = []
    for name in task_names:
        task_dir = topk_base / name
        if task_has_kernel(task_dir):
            completed.append(name)
        else:
            missing.append(name)
    return {
        "exp_root": str(exp_root),
        "topk_base": str(topk_base),
        "total": len(task_names),
        "completed": len(completed),
        "missing": missing,
        "completed_names": completed,
        "has_outputs": True,
    }


def main():
    definitions_dir = DEFINITIONS_DIR
    if not definitions_dir.is_dir():
        print(f"definitions 目录不存在: {definitions_dir}", file=sys.stderr)
        sys.exit(1)

    task_names = get_definition_task_names()
    print(f"definitions/topk 任务数: {len(task_names)}")
    print("任务列表:", task_names)
    print()

    # 要检查的实验目录（可配置）
    exp_roots = [
        KERNELEVAL_EXP / "GLM-5" / "glm-5-topk-20260309",
        KERNELEVAL_EXP / "GLM-4.7" / "GLM-4.7-topk-20260308",
        KERNELEVAL_EXP / "MiniMax-m2.5" / "minimax-m2.5-topk-20260309",
        KERNELEVAL_EXP / "MiniMax-m2.5" / "minimax-m2.5-topk-20260309-2",
    ]

    results = []
    for exp_root in exp_roots:
        if not exp_root.is_dir():
            print(f"跳过（不存在）: {exp_root}")
            continue
        r = check_experiment(exp_root, task_names)
        results.append(r)

    # 合并两次 MiniMax 实验：只要在任一次中有 kernel 即视为该任务完成
    minimax_1 = next((r for r in results if "minimax-m2.5-topk-20260309" in r["exp_root"] and "minimax-m2.5-topk-20260309-2" not in r["exp_root"]), None)
    minimax_2 = next((r for r in results if "minimax-m2.5-topk-20260309-2" in r["exp_root"]), None)
    if minimax_1 and minimax_2:
        merged_completed = set(minimax_1.get("completed_names", [])) | set(minimax_2.get("completed_names", []))
        merged_missing = [n for n in task_names if n not in merged_completed]
        results.append({
            "exp_root": "[合并] MiniMax-m2.5 topk (20260309 + 20260309-2)",
            "topk_base": "(合并)",
            "total": len(task_names),
            "completed": len(merged_completed),
            "missing": merged_missing,
            "completed_names": list(merged_completed),
            "has_outputs": True,
        })

    # 打印报告
    print("=" * 70)
    print("各 topk 实验完整性（以 definitions/topk 为准）")
    print("=" * 70)
    for r in results:
        exp_name = Path(r["exp_root"]).name if not r["exp_root"].startswith("[") else r["exp_root"]
        print(f"\n实验: {exp_name}")
        if not r["has_outputs"]:
            print("  无 outputs/topk → 不完整")
            continue
        print(f"  topk 输出根: {r['topk_base']}")
        print(f"  已完成: {r['completed']}/{r['total']}")
        if r["missing"]:
            print(f"  缺失任务: {r['missing']}")
        else:
            print("  结论: 完整")
    print("\n" + "=" * 70)
    print("汇总（合并后 MiniMax 单独一行）")
    print("=" * 70)
    for r in results:
        exp_name = Path(r["exp_root"]).name if not r["exp_root"].startswith("[") else r["exp_root"]
        status = "完整" if r["has_outputs"] and not r["missing"] else "不完整"
        print(f"  {exp_name}: {status} ({r['completed']}/{r['total']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
