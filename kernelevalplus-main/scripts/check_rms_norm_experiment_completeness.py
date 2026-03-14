#!/usr/bin/env python3
"""
根据 kernelevalplus/definitions/rms_norm 下的定义，检查 rms_norm 实验目录是否完整。
完整 = 每个定义对应的任务目录存在且含有 kernel 文件（.cu）。
"""

import json
import sys
from pathlib import Path

DEFINITIONS_DIR = Path(__file__).resolve().parents[1] / "definitions" / "rms_norm"
KERNELEVAL_EXP = Path("/home/qinhaiyan/KERNELEVAL-exp")


def get_definition_task_names() -> list[str]:
    """从 definitions/rms_norm 下所有 JSON 的 name 字段收集任务名（唯一、有序）。"""
    names = []
    seen = set()
    for j in sorted(DEFINITIONS_DIR.rglob("*.json")):
        with open(j, "r", encoding="utf-8") as f:
            d = json.load(f)
        n = d.get("name", "").strip()
        if not n:
            continue
        # 部分定义里 name 误带了 .json 后缀，统一去掉以便和实验目录名对齐
        if n.endswith(".json"):
            n = n[:-5]
        if n not in seen:
            seen.add(n)
            names.append(n)
    return names


def find_rms_norm_output_base(exp_root: Path) -> Path | None:
    """
    在实验根目录下找到 rms_norm 任务目录的根。
    - GLM-4.7: exp_root 即 rms_norm 根，下面直接是 fp32_rms_norm_* 目录。
    - GLM-5: exp_root/outputs 下直接是 fp32_rms_norm_* 目录。
    """
    if not exp_root.is_dir():
        return None
    # 直接检查当前目录下是否有 fp32_rms_norm_ 开头的子目录
    for sub in exp_root.iterdir():
        if sub.is_dir() and sub.name.startswith("fp32_rms_norm_"):
            return exp_root
    # outputs 下直接是 fp32_rms_norm_*（如 GLM-5 rms_norm-glm-5-20260311）
    outputs = exp_root / "outputs"
    if outputs.is_dir():
        for sub in outputs.iterdir():
            if sub.is_dir() and sub.name.startswith("fp32_rms_norm_"):
                return outputs
        rn = outputs / "rms_norm"
        if rn.is_dir():
            return rn
        for sub in outputs.iterdir():
            if sub.is_dir():
                r = sub / "rms_norm"
                if r.is_dir():
                    return r
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
    base = find_rms_norm_output_base(exp_root)
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
    if not DEFINITIONS_DIR.is_dir():
        print(f"definitions 目录不存在: {DEFINITIONS_DIR}", file=sys.stderr)
        sys.exit(1)

    task_names = get_definition_task_names()
    print(f"definitions/rms_norm 任务数: {len(task_names)}")
    print("任务列表:", task_names)
    print()

    exp_roots = [
        KERNELEVAL_EXP / "GLM-4.7" / "rms_norm",
        KERNELEVAL_EXP / "GLM-5" / "rms_norm-glm-5-20260311",
        KERNELEVAL_EXP / "MiniMax-m2.5" / "minimax-m2.5-rms_norm-20260309",
        # 如有其他模型的 rms_norm 实验可在此添加
    ]

    results = []
    for exp_root in exp_roots:
        if not exp_root.is_dir():
            print(f"跳过（不存在）: {exp_root}")
            continue
        r = check_experiment(exp_root, task_names)
        results.append(r)

    print("=" * 70)
    print("rms_norm 实验完整性（以 definitions/rms_norm 为准）")
    print("=" * 70)
    for r in results:
        exp_name = Path(r["exp_root"]).name
        print(f"\n实验: {exp_name}")
        if not r["has_outputs"]:
            print("  未找到 rms_norm 任务根目录 → 不完整")
            continue
        print(f"  任务根: {r['output_base']}")
        print(f"  已完成: {r['completed']}/{r['total']}")
        if r["missing"]:
            print(f"  缺失任务: {r['missing']}")
        else:
            print("  结论: 完整")
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
