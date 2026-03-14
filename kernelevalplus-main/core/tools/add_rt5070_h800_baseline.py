#!/usr/bin/env python3
"""
将 RTX5070 和 H800 的 GGML baseline 数据添加到 baseline_data_compact.json
"""
import json
import re
from pathlib import Path
from collections import defaultdict

# GGML 结果目录
GGML_RESULTS_DIR = Path("/home/qinhaiyan/ggml-python/results")

# Baseline 文件
BASELINE_FILE = Path("/home/qinhaiyan/kernelevalplus/data/baseline/baseline_data_compact.json")

# 输出文件
OUTPUT_FILE = Path("/home/qinhaiyan/kernelevalplus/data/baseline/baseline_data_compact.json")


def parse_tflops(s):
    """从字符串中提取 TFLOPS 值，例如 '3.80 TFLOPS' -> 3.80"""
    match = re.search(r'([\d.]+)\s*TFLOPS', str(s))
    if match:
        return float(match.group(1))
    return None


def parse_us_per_run(s):
    """从字符串中提取 us/run 值"""
    match = re.search(r'([\d.]+)\s*us/run', str(s))
    if match:
        return float(match.group(1))
    return None


def convert_tflops_to_gflops(tflops):
    """转换 TFLOPS 到 GFLOPS"""
    return tflops * 1000 if tflops else None


def process_raw_results_json(file_path, hardware_name):
    """处理 raw_results.json 格式 (RTX5070)"""
    with open(file_path) as f:
        data = json.load(f)

    # 按 (quant_type, m, n, k) 分组，取 GFLOPS 最高值
    results = {}

    for entry in data:
        quant_type = entry.get("quant_type", "")
        m = entry.get("m")
        n = entry.get("n")
        k = entry.get("k")

        if not all([quant_type, m, n, k]):
            continue

        timing = entry.get("timing", {})
        tflops = timing.get("tflops")
        us_per_run = timing.get("compute_ms")

        key = (quant_type.lower(), m, n, k)

        if key not in results or (tflops and tflops > results[key].get("tflops", 0)):
            results[key] = {
                "tflops": tflops,
                "gflops": tflops * 1000 if tflops else None,
                "us_per_run": us_per_run * 1000 if us_per_run else None,
            }

    return results


def process_gemm_perf_txt(file_path, hardware_name):
    """处理 gemm_perf_result.txt 格式 (H800)"""
    results = {}

    with open(file_path) as f:
        for line in f:
            # 匹配 MUL_MAT 格式
            # MUL_MAT(type_a=q4_0,type_b=f32,m=4096,n=1,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):
            # 113278 runs -     8.83 us/run -  33.55 MFLOP/run - [1;34m  3.80 TFLOPS[0m
            # 简化正则，分步匹配
            mul_match = re.search(r'MUL_MAT\(type_a=([^,]+),type_b=([^,]+),m=(\d+),n=(\d+),k=(\d+)', line)
            if not mul_match:
                continue

            type_a = mul_match.group(1)
            type_b = mul_match.group(2)
            m = int(mul_match.group(3))
            n = int(mul_match.group(4))
            k = int(mul_match.group(5))

            # 匹配性能数据
            # 格式: 113278 runs -     8.83 us/run -  33.55 MFLOP/run - [1;34m  3.80 TFLOPS[0m
            perf_match = re.search(r'(\d+) runs -\s+([\d.]+) us/run', line)
            if not perf_match:
                continue

            us_per_run = float(perf_match.group(2))

            # 匹配 TFLOPS
            tflops = None
            tflops_match = re.search(r'([\d.]+)\s*TFLOPS', line)
            if tflops_match:
                tflops = float(tflops_match.group(1))
            else:
                # 尝试匹配 MFLOP
                mflop_match = re.search(r'([\d.]+)\s*MFLOP', line)
                if mflop_match:
                    tflops = float(mflop_match.group(1)) / 1000
                else:
                    # 尝试匹配 GFLOP
                    gflop_match = re.search(r'([\d.]+)\s*GFLOP', line)
                    if gflop_match:
                        tflops = float(gflop_match.group(1)) / 1000

            key = (type_a.lower(), m, n, k)

            if key not in results or (tflops and tflops > results[key].get("tflops", 0)):
                results[key] = {
                    "tflops": tflops,
                    "gflops": tflops * 1000 if tflops else None,
                    "us_per_run": us_per_run,
                }

    return results


def get_case_id(type_a, type_b, m, n, k):
    """生成 baseline case ID"""
    # GGML 格式: w4a32c8_q4_0_f32_m{M}_n{N}_k{K}
    # 其中 type_b 决定前缀: q4_0 -> w4a32c8, q8_0 -> w8a32c8
    prefix = "w4a32c8" if type_a.lower() == "q4_0" else "w8a32c8"
    return f"{prefix}_{type_a.lower()}_{type_b.lower()}_m{m}_n{n}_k{k}"


def main():
    # 1. 加载现有 baseline 数据
    with open(BASELINE_FILE) as f:
        baseline_data = json.load(f)

    # 2. 处理 RTX5070 数据 (raw_results.json)
    rtx5070_dir = GGML_RESULTS_DIR / "nvidia-RTX5070-Laptop"
    rtx5070_raw = rtx5070_dir / "raw_results.json"

    if rtx5070_raw.exists():
        print(f"Processing RTX5070 data from {rtx5070_raw}")
        rtx5070_results = process_raw_results_json(rtx5070_raw, "RTX5070")

        # 添加到 baseline 数据
        for (type_a, m, n, k), perf in rtx5070_results.items():
            case_id = get_case_id(type_a, "f32", m, n, k)

            if case_id not in baseline_data:
                baseline_data[case_id] = {
                    "type_a": type_a.lower(),
                    "type_b": "f32",
                    "m": m,
                    "n": n,
                    "k": k,
                    "hardware": {}
                }

            baseline_data[case_id]["hardware"]["RTX5070"] = {
                "tflops": round(perf["tflops"], 2) if perf.get("tflops") else None,
                "gflops": round(perf["gflops"], 2) if perf.get("gflops") else None,
                "us_per_run": round(perf["us_per_run"], 2) if perf.get("us_per_run") else None,
            }

        print(f"  Added {len(rtx5070_results)} RTX5070 entries")
    else:
        print(f"Warning: {rtx5070_raw} not found")

    # 3. 处理 H800 数据 (gemm_perf_result.txt)
    h800_dir = GGML_RESULTS_DIR / "nvidia-H800-80G"
    h800_txt = h800_dir / "gemm_perf_result.txt"

    if h800_txt.exists():
        print(f"Processing H800 data from {h800_txt}")
        h800_results = process_gemm_perf_txt(h800_txt, "H800")

        # 添加到 baseline 数据
        for (type_a, m, n, k), perf in h800_results.items():
            case_id = get_case_id(type_a, "f32", m, n, k)

            if case_id not in baseline_data:
                baseline_data[case_id] = {
                    "type_a": type_a.lower(),
                    "type_b": "f32",
                    "m": m,
                    "n": n,
                    "k": k,
                    "hardware": {}
                }

            baseline_data[case_id]["hardware"]["H800"] = {
                "tflops": round(perf["tflops"], 2) if perf.get("tflops") else None,
                "gflops": round(perf["gflops"], 2) if perf.get("gflops") else None,
                "us_per_run": round(perf["us_per_run"], 2) if perf.get("us_per_run") else None,
            }

        print(f"  Added {len(h800_results)} H800 entries")
    else:
        print(f"Warning: {h800_txt} not found")

    # 4. 处理 RTX4070 Laptop 数据 (如果存在)
    rtx4070_dir = GGML_RESULTS_DIR / "nvidia-RTX4070-Laptop"
    rtx4070_raw = rtx4070_dir / "raw_results.json"

    if rtx4070_raw.exists():
        print(f"Processing RTX4070 Laptop data from {rtx4070_raw}")
        rtx4070_results = process_raw_results_json(rtx4070_raw, "RTX4070")

        for (type_a, m, n, k), perf in rtx4070_results.items():
            case_id = get_case_id(type_a, "f32", m, n, k)

            if case_id not in baseline_data:
                baseline_data[case_id] = {
                    "type_a": type_a.lower(),
                    "type_b": "f32",
                    "m": m,
                    "n": n,
                    "k": k,
                    "hardware": {}
                }

            baseline_data[case_id]["hardware"]["RTX4070"] = {
                "tflops": round(perf["tflops"], 2) if perf.get("tflops") else None,
                "gflops": round(perf["gflops"], 2) if perf.get("gflops") else None,
                "us_per_run": round(perf["us_per_run"], 2) if perf.get("us_per_run") else None,
            }

        print(f"  Added {len(rtx4070_results)} RTX4070 Laptop entries")
    else:
        print(f"Info: {rtx4070_raw} not found")

    # 5. 保存更新后的 baseline 数据
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\nSaved updated baseline to {OUTPUT_FILE}")
    print(f"Total entries: {len(baseline_data)}")


if __name__ == "__main__":
    main()
