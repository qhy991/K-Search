#!/usr/bin/env python3
"""
根据 kernelevalplus/definitions/quant_gemm 下的定义，检查 quant_gemm 实验目录是否完整。
完整 = 每个定义对应的任务目录存在且含有 kernel 文件（.cu）。
"""

import json
import sys
from pathlib import Path

DEFINITIONS_DIR = Path(__file__).resolve().parents[1] / "definitions" / "quant_gemm"


def get_definition_task_names() -> list[str]:
    """从 definitions/quant_gemm 下所有 JSON 的 name 字段收集任务名（唯一、有序）。"""
    names = []
    seen = set()
    for j in sorted(DEFINITIONS_DIR.rglob("*.json")):
        with open(j, "r", encoding="utf-8") as f:
            d = json.load(f)
        n = d.get("name", "").strip()
        if not n:
            continue
        if n.endswith(".json"):
            n = n[:-5]
        if n not in seen:
            seen.add(n)
            names.append(n)
    return names


def find_quant_gemm_output_base(exp_root: Path) -> Path | None:
    """找到 quant_gemm 任务目录的根：实验根本身、outputs、或 outputs/quant_gemm。"""
    if not exp_root.is_dir():
        return None
    # 实验根下直接是任务目录（如 GLM-5/quant_gemm）
    for sub in exp_root.iterdir():
        if sub.is_dir() and (sub.name.startswith("w4a32c8_") or sub.name.startswith("w8a32c8_")):
            return exp_root
    outputs = exp_root / "outputs"
    if not outputs.is_dir():
        return None
    for sub in outputs.iterdir():
        if sub.is_dir() and (sub.name.startswith("w4a32c8_") or sub.name.startswith("w8a32c8_")):
            return outputs
    qg = outputs / "quant_gemm"
    if qg.is_dir():
        return qg
    for sub in outputs.iterdir():
        if sub.is_dir() and (sub / "quant_gemm").is_dir():
            return sub / "quant_gemm"
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
    """检查一个实验的完整性。"""
    base = find_quant_gemm_output_base(exp_root)
    if not base:
        return {
            "exp_root": str(exp_root),
            "output_base": None,
            "total": len(task_names),
            "completed": 0,
            "missing": list(task_names),
            "completed_names": [],
            "has_outputs": False,
        }
    completed = []
    missing = []
    for name in task_names:
        task_dir = base / name
        if task_has_kernel(task_dir):
            completed.append(name)
        else:
            missing.append(name)
    return {
        "exp_root": str(exp_root),
        "output_base": str(base),
        "total": len(task_names),
        "completed": len(completed),
        "missing": missing,
        "completed_names": completed,
        "has_outputs": True,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python3 check_quant_gemm_experiment_completeness.py <exp_root>")
        print("例:   python3 check_quant_gemm_experiment_completeness.py /path/to/quant_gemm-glm-5-20260311")
        return 1
    exp_root = Path(sys.argv[1]).resolve()
    if not DEFINITIONS_DIR.is_dir():
        print(f"definitions 目录不存在: {DEFINITIONS_DIR}", file=sys.stderr)
        return 1

    task_names = get_definition_task_names()
    print(f"definitions/quant_gemm 任务数: {len(task_names)}")
    print()

    r = check_experiment(exp_root, task_names)
    exp_name = exp_root.name
    print("=" * 70)
    print(f"quant_gemm 实验完整性: {exp_name}")
    print("=" * 70)
    if not r["has_outputs"]:
        print("未找到 quant_gemm 输出根（outputs 下无 w4a32c8_* / w8a32c8_* 任务目录）")
        return 1
    print(f"任务根: {r['output_base']}")
    print(f"已完成: {r['completed']}/{r['total']} ({100*r['completed']//r['total']}%)")
    if r["completed_names"]:
        print("\n已完成任务:")
        for n in r["completed_names"]:
            print(f"  - {n}")
    if r["missing"]:
        print(f"\n缺失任务 ({len(r['missing'])} 个):")
        for n in r["missing"][:30]:
            print(f"  - {n}")
        if len(r["missing"]) > 30:
            print(f"  ... 及另外 {len(r['missing'])-30} 个")
    else:
        print("\n结论: 完整")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
