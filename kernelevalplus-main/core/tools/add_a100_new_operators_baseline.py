#!/usr/bin/env python3
"""
将 A100 的 Flash Attention、RMS Norm、TopK baseline 数据添加到系统

每个算子类型单独存储一个文件：
- flash_attn_baseline.json
- rms_norm_baseline.json
- topk_baseline.json
"""
import json
import re
from pathlib import Path
from collections import defaultdict

# GGML 结果目录
GGML_RESULTS_DIR = Path("/home/qinhaiyan/ggml-python/results")

# 输出目录
OUTPUT_DIR = Path("/home/qinhaiyan/kernelevalplus/data/baseline")


def parse_flash_attn_md(md_file):
    """解析 Flash Attention Markdown 报告"""
    results = {}

    with open(md_file) as f:
        content = f.read()

    lines = content.split('\n')
    current_model = None
    current_kv_type = None
    current_cache_size = None

    for i, line in enumerate(lines):
        # 检测模型 (### ModelName)
        model_match = re.match(r'###\s+([A-Z]\w+[0-9.-]+[Bb]?)', line)
        if model_match:
            current_model = model_match.group(1)
            current_kv_type = None
            current_cache_size = None
            continue

        # 检测 KV 类型 (#### type_KV=XXX)
        kv_match = re.match(r'####\s+type_KV=(\S+)', line)
        if kv_match:
            current_kv_type = kv_match.group(1)
            current_cache_size = None
            continue

        # 检测 cache size (##### kv_cache_size=XXX)
        cache_match = re.match(r'#####\s+kv_cache_size=(\d+)', line)
        if cache_match:
            current_cache_size = int(cache_match.group(1))
            continue

        # 解析数据行
        # 格式: | 1 | 57,330 | 20.04 | 8.39 MFLOP/run | **0.42** |
        data_match = re.match(r'\|\s*(\d+)\s*\|\s*[\d,]+\s*\|\s*([\d.]+)\s*\|.*\|\s*\*\*([\d.]+)\*\*', line)
        if data_match and current_model and current_kv_type and current_cache_size:
            nb = int(data_match.group(1))
            us_per_run = float(data_match.group(2))
            tflops = float(data_match.group(3))

            key = (current_model, current_kv_type, current_cache_size, nb)

            results[key] = {
                "model": current_model,
                "kv_type": current_kv_type,
                "kv_cache_size": current_cache_size,
                "nb": nb,
                "tflops": tflops,
                "us_per_run": us_per_run
            }

    return results


def parse_rms_norm_md(md_file):
    """解析 RMS Norm Markdown 报告"""
    results = {}

    with open(md_file) as f:
        lines = f.readlines()

    current_hidden_size = None

    for line in lines:
        # 检测 hidden size (### Hidden Size = XXX)
        hs_match = re.match(r'###\s+Hidden Size\s*=\s*(\d+)', line)
        if hs_match:
            current_hidden_size = int(hs_match.group(1))
            continue

        # 解析数据行
        # 格式: | [128,8,1,1] | 4.0 KB | 352,213 | 2.89 | 8 | **2.64** |
        data_match = re.match(r'\|\s*\[(\d+),(\d+),(\d+),(\d+)\]\s*\|\s*[\d.]+\s+\w+\s*\|\s*[\d,]+\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*\*\*([\d.]+)\*\*', line)
        if data_match and current_hidden_size:
            ne0, ne1, ne2, ne3 = map(int, data_match.groups()[:4])
            us_per_run = float(data_match.group(5))
            gbps = float(data_match.group(6))

            key = (current_hidden_size, ne0, ne1, ne2, ne3)

            results[key] = {
                "hidden_size": current_hidden_size,
                "ne": [ne0, ne1, ne2, ne3],
                "us_per_run": us_per_run,
                "gbps": gbps
            }

    return results


def parse_topk_md(md_file):
    """解析 TopK Markdown 报告"""
    results = {}

    with open(md_file) as f:
        lines = f.readlines()

    current_k = None
    current_ne0 = None

    for line in lines:
        # 检测 k 值 (### k = XX)
        k_match = re.match(r'###\s+k\s*=\s*(\d+)', line)
        if k_match:
            current_k = int(k_match.group(1))
            current_ne0 = None
            continue

        # 检测 ne0 值 (#### ne0 = XX)
        ne0_match = re.match(r'####\s+ne0\s*=\s*(\d+)', line)
        if ne0_match:
            current_ne0 = int(ne0_match.group(1))
            continue

        # 解析数据行
        # 格式: | [160,1,1,1] | 6 | 0.6 KB | 65,536 | 17.04 | 0 | **0.04** |
        data_match = re.match(r'\|\s*\[(\d+),(\d+),(\d+),(\d+)\]\s*\|\s*\d+\s*\|\s*[\d.]+\s+\w+\s*\|\s*[\d,]+\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*\*\*([\d.]+)\*\*', line)
        if data_match and current_k and current_ne0:
            ne = list(map(int, data_match.groups()[:4]))
            us_per_run = float(data_match.group(5))
            gbps = float(data_match.group(6))

            key = (current_k, current_ne0, ne[0], ne[1], ne[2], ne[3])

            results[key] = {
                "k": current_k,
                "ne0": current_ne0,
                "ne": ne,
                "us_per_run": us_per_run,
                "gbps": gbps
            }

    return results


def build_flash_attn_case_id(model, kv_type, kv_cache_size, nb):
    """构建 Flash Attention case ID"""
    return f"flash_attn_{model}_{kv_type}_cache{kv_cache_size}_nb{nb}"


def build_rms_norm_case_id(hidden_size, ne):
    """构建 RMS Norm case ID"""
    ne_str = "x".join(map(str, ne))
    return f"rms_norm_hs{hidden_size}_{ne_str}"


def build_topk_case_id(k, ne0, ne):
    """构建 TopK case ID"""
    ne_str = "x".join(map(str, ne))
    return f"topk_k{k}_ne0{ne0}_{ne_str}"


def main():
    a100_dir = GGML_RESULTS_DIR / "nvidia-A100-SXM4-80G"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_count = 0

    # ==================== Flash Attention ====================
    flash_attn_md = a100_dir / "flash_attn_perf_result.md"
    flash_attn_file = OUTPUT_DIR / "flash_attn_baseline.json"
    flash_attn_data = {}

    if flash_attn_md.exists():
        print(f"Parsing Flash Attention from {flash_attn_md}")
        flash_results = parse_flash_attn_md(flash_attn_md)

        for key, data in flash_results.items():
            case_id = build_flash_attn_case_id(
                data["model"], data["kv_type"],
                data["kv_cache_size"], data["nb"]
            )

            flash_attn_data[case_id] = {
                "model": data["model"],
                "kv_type": data["kv_type"],
                "kv_cache_size": data["kv_cache_size"],
                "nb": data["nb"],
                "hardware": {
                    "A100": {
                        "tflops": round(data["tflops"], 2),
                        "us_per_run": round(data["us_per_run"], 2) if data.get("us_per_run") else None
                    }
                }
            }

        # 保存到独立文件
        with open(flash_attn_file, 'w') as f:
            json.dump(flash_attn_data, f, indent=2)

        print(f"  Saved {len(flash_attn_data)} entries to {flash_attn_file.name}")
        total_count += len(flash_attn_data)
    else:
        print(f"Warning: {flash_attn_md} not found")

    # ==================== RMS Norm ====================
    rms_norm_md = a100_dir / "rms_norm_perf_result.md"
    rms_norm_file = OUTPUT_DIR / "rms_norm_baseline.json"
    rms_norm_data = {}

    if rms_norm_md.exists():
        print(f"Parsing RMS Norm from {rms_norm_md}")
        rms_results = parse_rms_norm_md(rms_norm_md)

        for key, data in rms_results.items():
            case_id = build_rms_norm_case_id(data["hidden_size"], data["ne"])

            rms_norm_data[case_id] = {
                "hidden_size": data["hidden_size"],
                "ne": data["ne"],
                "hardware": {
                    "A100": {
                        "gbps": round(data["gbps"], 2),
                        "us_per_run": round(data["us_per_run"], 2)
                    }
                }
            }

        # 保存到独立文件
        with open(rms_norm_file, 'w') as f:
            json.dump(rms_norm_data, f, indent=2)

        print(f"  Saved {len(rms_norm_data)} entries to {rms_norm_file.name}")
        total_count += len(rms_norm_data)
    else:
        print(f"Warning: {rms_norm_md} not found")

    # ==================== TopK ====================
    topk_md = a100_dir / "topk_perf_result.md"
    topk_file = OUTPUT_DIR / "topk_baseline.json"
    topk_data = {}

    if topk_md.exists():
        print(f"Parsing TopK from {topk_md}")
        topk_results = parse_topk_md(topk_md)

        for key, data in topk_results.items():
            case_id = build_topk_case_id(data["k"], data["ne0"], data["ne"])

            topk_data[case_id] = {
                "k": data["k"],
                "ne0": data["ne0"],
                "ne": data["ne"],
                "hardware": {
                    "A100": {
                        "gbps": round(data["gbps"], 2),
                        "us_per_run": round(data["us_per_run"], 2)
                    }
                }
            }

        # 保存到独立文件
        with open(topk_file, 'w') as f:
            json.dump(topk_data, f, indent=2)

        print(f"  Saved {len(topk_data)} entries to {topk_file.name}")
        total_count += len(topk_data)
    else:
        print(f"Warning: {topk_md} not found")

    print(f"\n{'='*60}")
    print(f"Total: {total_count} entries saved to {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"Generated files:")
    for f in OUTPUT_DIR.glob("*_baseline.json"):
        if f != "baseline_data_compact.json":
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
