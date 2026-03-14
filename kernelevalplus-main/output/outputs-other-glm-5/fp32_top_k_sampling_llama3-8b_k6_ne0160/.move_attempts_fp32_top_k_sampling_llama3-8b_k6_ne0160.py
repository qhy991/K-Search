#!/usr/bin/env python3
"""
Script to organize attempts for fp32_top_k_sampling_llama3-8b_k6_ne0160
Moves all kernel versions from experiments to output directory
"""

import os
import shutil
from pathlib import Path

# Define paths
OUTPUT_DIR = Path("/home/qinhaiyan/kernelevalplus/output/outputs-other-glm-5/fp32_top_k_sampling_llama3-8b_k6_ne0160")
EXPERIMENTS_DIR = Path("/home/qinhaiyan/kernelevalplus/experiments/fp32_top_k_sampling_llama3_8b_k6_ne0160")

def main():
    # Create output directory structure
    attempts_dir = OUTPUT_DIR / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Experiments directory: {EXPERIMENTS_DIR}")

    # Copy best kernel
    best_kernel = EXPERIMENTS_DIR / "kernel_best.cu"
    if best_kernel.exists():
        shutil.copy2(best_kernel, OUTPUT_DIR / "kernel_best.cu")
        print(f"Copied: {best_kernel.name}")

    # Copy all versions
    versions_dir = EXPERIMENTS_DIR / "versions"
    if versions_dir.exists():
        for kernel_file in versions_dir.glob("*.cu"):
            shutil.copy2(kernel_file, attempts_dir / kernel_file.name)
            print(f"Copied version: {kernel_file.name}")

    # Copy test results
    test_results_dir = EXPERIMENTS_DIR / "test_results"
    if test_results_dir.exists():
        output_test_dir = OUTPUT_DIR / "test_results"
        output_test_dir.mkdir(exist_ok=True)
        for result_file in test_results_dir.glob("*"):
            shutil.copy2(result_file, output_test_dir / result_file.name)
            print(f"Copied test result: {result_file.name}")

    # Copy summary
    summary_file = EXPERIMENTS_DIR / "SUMMARY.md"
    if summary_file.exists():
        shutil.copy2(summary_file, OUTPUT_DIR / "SUMMARY_old.md")
        print(f"Copied: SUMMARY.md -> SUMMARY_old.md")

    print("\nOrganization complete!")
    print(f"\nOutput structure:")
    for item in sorted(OUTPUT_DIR.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(OUTPUT_DIR)
            print(f"  {rel_path}")

if __name__ == "__main__":
    main()
