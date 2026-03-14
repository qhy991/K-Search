#!/usr/bin/env python3
"""Organize attempts for w4a32c8_q4_0_fp32_int8_ds2_moe_up_n12288_k5120"""
import json
import os
import shutil
from pathlib import Path

BASE_DIR = Path("/home/qinhaiyan/kernelevalplus")
OUTPUT_DIR = BASE_DIR / "output/outputs-quant_gemm-GLM-4.7/w4a32c8_q4_0_fp32_int8_ds2_moe_up_n12288_k5120"
ATTEMPTS_DIR = BASE_DIR / "attempts"
PROBLEM_NAME = "w4a32c8_q4_0_fp32_int8_ds2_moe_up_n12288_k5120"

attempts_output_dir = OUTPUT_DIR / "attempts"
attempts_output_dir.mkdir(exist_ok=True, parents=True)

print("="*80)
print(f"  Organizing Attempts: {PROBLEM_NAME}")
print("="*80)
print()

version_summary = []
best_per_config = {"M=1": {"ver": None, "tflops": 0}, "M=8": {"ver": None, "tflops": 0}, "M=512": {"ver": None, "tflops": 0}}

for v in range(1, 9):
    attempt_dir = ATTEMPTS_DIR / f"{PROBLEM_NAME}_v{v}"
    if not attempt_dir.exists():
        continue
    
    results_file = attempt_dir / "test_results.json"
    if not results_file.exists():
        continue
    
    with open(results_file) as f:
        results = json.load(f)
    
    passed = results.get("correctness", {}).get("passed", False)
    perf = results.get("performance", {}).get("benchmarks", [])
    
    perf_data = {}
    for bench in perf:
        tflops = bench["tflops"]
        if bench["config"] == "single_token":
            perf_data["M=1"] = tflops
            if tflops > best_per_config["M=1"]["tflops"]:
                best_per_config["M=1"] = {"ver": f"v{v}", "tflops": tflops}
        elif bench["config"] == "small_batch":
            perf_data["M=8"] = tflops
            if tflops > best_per_config["M=8"]["tflops"]:
                best_per_config["M=8"] = {"ver": f"v{v}", "tflops": tflops}
        elif bench["config"] == "large_batch":
            perf_data["M=512"] = tflops
            if tflops > best_per_config["M=512"]["tflops"]:
                best_per_config["M=512"] = {"ver": f"v{v}", "tflops": tflops}
    
    version_summary.append({"version": f"v{v}", "correctness": "PASS" if passed else "FAIL",
                          "M=1": perf_data.get("M=1", 0), "M=8": perf_data.get("M=8", 0), "M=512": perf_data.get("M=512", 0)})
    
    version_output_dir = attempts_output_dir / f"v{v}"
    version_output_dir.mkdir(exist_ok=True)
    
    kernel_src = attempt_dir / "kernel.cu"
    if kernel_src.exists():
        shutil.copy2(kernel_src, version_output_dir / "kernel.cu")
    shutil.copy2(results_file, version_output_dir / "test_results.json")
    
    print(f"v{v}: {'PASS' if passed else 'FAIL':4} | M=1:{perf_data.get('M=1',0):5.2f} | M=8:{perf_data.get('M=8',0):5.2f} | M=512:{perf_data.get('M=512',0):5.2f}")

print()
print("="*80)
print("  Best Performance by Configuration")
print("="*80)
for config, data in best_per_config.items():
    if data["ver"]:
        print(f"  {config}: {data['ver']} with {data['tflops']:.3f} TFLOPS")

index_file = attempts_output_dir / "INDEX.md"
with open(index_file, "w") as f:
    f.write("# Attempts Index\n\n## Version Summary\n\n")
    f.write("| Version | M=1 | M=8 | M=512 | Status |\n")
    f.write("|----------|-----|-----|------|--------|\n")
    for v in version_summary:
        f.write(f"| {v['version']:8} | {v['M=1']:5.2f} | {v['M=8']:5.2f} | {v['M=512']:6.2f} | {v['correctness']:4} |\n")
    f.write("\n## Best by Configuration\n\n")
    for config, data in best_per_config.items():
        if data["ver"]:
            f.write(f"- **{config}**: {data['ver']} with {data['tflops']:.3f} TFLOPS\n")

print()
print(f"  Complete! All attempts organized in: attempts/")
print("="*80)
