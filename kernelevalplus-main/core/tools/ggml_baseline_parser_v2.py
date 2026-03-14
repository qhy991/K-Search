#!/usr/bin/env python3
"""
GGML 性能结果解析器 V2

支持两种数据格式：
1. 旧格式：gemm_perf_result.txt (RTX4090, A100, RTX4070)
2. 新格式：raw_results.json (RTX5070 Laptop 等，支持多个 batch size)

Usage:
    # 解析所有结果
    python -m python.tools.ggml_baseline_parser_v2 --results-dir /home/haiyan/Agent4Kernel/ggml-python/results
"""

import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import hashlib


@dataclass
class PerformanceRecord:
    """单条性能记录"""
    hardware: str
    type_a: str  # q4_0, q4_1, q8_0
    type_b: str  # f32
    m: int
    n: int
    k: int
    runs: int
    us_per_run: float
    flop_per_run: float
    tflops: float
    flop_unit: str  # "MFLOP" or "GFLOP"

    @staticmethod
    def _get_weight_bits(type_a: str) -> int:
        """从量化类型获取 weight 位宽"""
        if type_a.startswith("q4"):
            return 4
        elif type_a.startswith("q5"):
            return 5
        elif type_a.startswith("q8"):
            return 8
        return 8  # 默认

    @staticmethod
    def _get_activation_bits(type_b: str) -> int:
        """从类型获取 activation 位宽"""
        if type_b == "f32" or type_b == "float32":
            return 32
        elif type_b == "f16" or type_b == "float16" or type_b == "fp16":
            return 16
        elif type_b.startswith("q8"):
            return 8
        return 32  # 默认

    @staticmethod
    def _get_compute_bits(type_a: str) -> int:
        """获取计算位宽 (GGML 都使用 INT8 DP4A 计算)"""
        return 8  # 所有 GGML 量化 GEMM 都使用 INT8 计算 (DP4A)

    def get_case_id(self) -> str:
        """
        生成唯一 case ID
        格式: w{weight}a{activation}c{compute}_{type_a}_{type_b}_m{M}_n{N}_k{K}
        例如: w4a32c8_q4_0_f32_m4096_n1_k4096
        """
        w_bits = self._get_weight_bits(self.type_a)
        a_bits = self._get_activation_bits(self.type_b)
        c_bits = self._get_compute_bits(self.type_a)

        return f"w{w_bits}a{a_bits}c{c_bits}_{self.type_a}_{self.type_b}_m{self.m}_n{self.n}_k{self.k}"


class GGMLBaselineParser:
    """GGML Baseline 解析器 V2"""

    # 硬件名称映射
    HARDWARE_MAPPING = {
        "nvidia-RTX5070-Laptop": "RTX5070",
        "nvidia-RTX4090-24G": "RTX4090",
        "nvidia-A100-SXM4-80G": "A100",
        "nvidia-A800-80GB-PCIe": "A800",
        "nvidia-RTX4070-Laptop": "RTX4070",
    }

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.records: List[PerformanceRecord] = []

    def parse_all(self) -> List[PerformanceRecord]:
        """解析所有硬件的结果"""
        all_records = []

        for hw_dir in self.results_dir.iterdir():
            if not hw_dir.is_dir():
                continue

            # 提取硬件名称
            hw_name = hw_dir.name
            hardware = self.HARDWARE_MAPPING.get(hw_name, hw_name)

            print(f"  📂 解析 {hw_name} -> {hardware}")

            # 尝试解析新格式 JSON
            json_file = hw_dir / "raw_results.json"
            if json_file.exists():
                records = self._parse_json_format(json_file, hardware)
                all_records.extend(records)
                print(f"     ✅ JSON 格式: {len(records)} 条记录")
                continue

            # 尝试解析旧格式文本
            txt_file = hw_dir / "gemm_perf_result.txt"
            if txt_file.exists():
                records = self._parse_text_format(txt_file, hardware)
                all_records.extend(records)
                print(f"     ✅ 文本格式: {len(records)} 条记录")
                continue

            print(f"     ⚠️  未找到结果文件")

        self.records = all_records
        return all_records

    def _parse_text_format(self, txt_file: Path, hardware: str) -> List[PerformanceRecord]:
        """解析旧格式文本 (gemm_perf_result.txt)"""
        records = []

        with open(txt_file) as f:
            lines = f.readlines()

        # 移除 ANSI 颜色代码
        ansi_escape = re.compile(r'\x1B\[[0-9;]*m')

        for line in lines:
            # 移除颜色代码
            line = ansi_escape.sub('', line)

            # 检查是否包含 MUL_MAT
            if 'MUL_MAT' not in line:
                continue

            # 提取 MUL_MAT 的参数（允许额外参数如 bs, nr, per, k_v, o）
            mulmat_match = re.search(r'MUL_MAT\(type_a=([a-z0-9_]+),type_b=([a-z0-9_]+),m=(\d+),n=(\d+),k=(\d+)', line)
            if not mulmat_match:
                continue

            type_a = mulmat_match.group(1)
            type_b = mulmat_match.group(2)
            m = int(mulmat_match.group(3))
            n = int(mulmat_match.group(4))
            k = int(mulmat_match.group(5))

            # 提取性能数据（支持 TFLOPS 和 GFLOPS 两种单位）
            perf_match = re.search(r'(\d+) runs\s+-\s+([\d.]+) us/run\s+-\s+([\d.]+)\s+([A-Z]+)LOP/run\s+-\s+([\d.]+)\s+([TG])FLOPS', line)
            if not perf_match:
                continue

            runs = int(perf_match.group(1))
            us_per_run = float(perf_match.group(2))
            flop_per_run = float(perf_match.group(3))
            flop_prefix = perf_match.group(4)  # M or G
            perf_value = float(perf_match.group(5))
            perf_unit = perf_match.group(6)  # T or G

            # 统一转换为 TFLOPS
            if perf_unit == "G":
                tflops = perf_value / 1000.0  # GFLOPS -> TFLOPS
            else:
                tflops = perf_value

            flop_unit = flop_prefix + "FLOP"

            record = PerformanceRecord(
                hardware=hardware,
                type_a=type_a,
                type_b=type_b,
                m=m,
                n=n,
                k=k,
                runs=runs,
                us_per_run=us_per_run,
                flop_per_run=flop_per_run,
                tflops=tflops,
                flop_unit=flop_unit
            )
            records.append(record)

        return records

    def _parse_json_format(self, json_file: Path, hardware: str) -> List[PerformanceRecord]:
        """解析新格式 JSON (raw_results.json)"""
        records = []

        with open(json_file) as f:
            data = json.load(f)

        for item in data:
            # 跳过失败的结果
            if item.get("status") != "OK":
                continue

            # 提取量化类型 (Q4_0 -> q4_0)
            # 保持下划线格式（不替换为点），与文本格式保持一致
            quant_type = item.get("quant_type", "").lower()
            if not quant_type:
                continue

            m = item.get("m", 1)
            n = item.get("n", 1)
            k = item.get("k", 1)

            # 提取性能数据
            timing = item.get("timing", {})
            compute_ms = timing.get("compute_ms", 0)
            tflops = timing.get("tflops", 0)

            if compute_ms > 0:
                us_per_run = compute_ms * 1000
                runs = int(1000 / compute_ms)  # 估算运行次数

                # 计算 FLOP
                total_ops = 2 * m * n * k  # multiply-add
                flop_per_run = total_ops / 1e6  # MFLOP
                flop_unit = "MFLOP" if flop_per_run < 1000 else "GFLOP"
                if flop_unit == "GFLOP":
                    flop_per_run = total_ops / 1e9

                record = PerformanceRecord(
                    hardware=hardware,
                    type_a=quant_type,
                    type_b="f32",
                    m=m,
                    n=n,
                    k=k,
                    runs=runs,
                    us_per_run=us_per_run,
                    flop_per_run=flop_per_run,
                    tflops=tflops,
                    flop_unit=flop_unit
                )
                records.append(record)

        return records

    def save_compact_format(self, output_file: str, merge: bool = False):
        """保存为紧凑格式 (baseline_data_compact.json)

        Args:
            output_file: 输出文件路径
            merge: 如果为 True，将新数据合并到现有文件中（保留未重新解析的硬件数据）
        """
        # 如果 merge 模式，先加载现有数据
        output_path = Path(output_file)
        if merge and output_path.exists():
            with open(output_path, 'r') as f:
                compact_data = json.load(f)
            print(f"  📂 合并模式: 加载现有 {len(compact_data)} 个 case")
        else:
            compact_data = {}

        # 按 case_id 组织新数据
        new_count = 0
        updated_count = 0
        for record in self.records:
            case_id = record.get_case_id()

            if case_id not in compact_data:
                compact_data[case_id] = {
                    "type_a": record.type_a,
                    "type_b": record.type_b,
                    "m": record.m,
                    "n": record.n,
                    "k": record.k,
                    "hardware": {}
                }
                new_count += 1

            # 添加/更新硬件性能数据
            hw_key = record.hardware
            if hw_key not in compact_data[case_id].get("hardware", {}):
                updated_count += 1
            compact_data[case_id]["hardware"][hw_key] = {
                "tflops": round(record.tflops, 2),
                "gflops": round(record.tflops * 1000, 1),
                "us_per_run": round(record.us_per_run, 2)
            }

        # 保存到文件
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(compact_data, f, indent=2)

        print(f"\n💾 已保存 {len(compact_data)} 个 case 到: {output_file}")
        if merge:
            print(f"  📊 新增 case: {new_count}, 新增硬件数据: {updated_count}")

        # 打印统计信息
        hardware_stats = {}
        for record in self.records:
            if record.hardware not in hardware_stats:
                hardware_stats[record.hardware] = set()
            hardware_stats[record.hardware].add((record.m, record.n, record.k))

        print(f"\n📊 硬件统计:")
        for hw, configs in sorted(hardware_stats.items()):
            print(f"   {hw}: {len(configs)} 个配置")

        # 打印 batch size 统计
        n_values = set(record.n for record in self.records)
        print(f"\n📏 Batch Size (N) 统计:")
        for n in sorted(n_values):
            count = sum(1 for r in self.records if r.n == n)
            print(f"   N={n}: {count} 条记录")

    def print_summary(self):
        """打印解析摘要"""
        print(f"\n{'='*60}")
        print(f"解析结果摘要")
        print(f"{'='*60}")
        print(f"总记录数: {len(self.records)}")

        # 按硬件统计
        hardware_count = {}
        for record in self.records:
            hardware_count[record.hardware] = hardware_count.get(record.hardware, 0) + 1

        print(f"\n按硬件统计:")
        for hw, count in sorted(hardware_count.items()):
            print(f"  {hw}: {count} 条记录")

        # 按类型统计
        type_count = {}
        for record in self.records:
            key = f"{record.type_a}_{record.type_b}"
            type_count[key] = type_count.get(key, 0) + 1

        print(f"\n按量化类型统计:")
        for dtype, count in sorted(type_count.items()):
            print(f"  {dtype}: {count} 条记录")


def main():
    parser = argparse.ArgumentParser(description="GGML Baseline 解析器 V2")
    parser.add_argument("--results-dir", required=True, help="GGML 结果目录")
    parser.add_argument("--output", default="data/baseline/baseline_data_compact.json", help="输出文件路径")
    parser.add_argument("--merge", action="store_true", help="合并模式：保留现有数据，只添加/更新新数据")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    print("="*60)
    print("GGML Baseline 解析器 V2")
    print("="*60)
    print(f"结果目录: {args.results_dir}")
    if args.merge:
        print(f"模式: 合并（保留现有数据）")

    # 创建解析器
    gp = GGMLBaselineParser(args.results_dir)

    # 解析所有结果
    print(f"\n🔍 开始解析...")
    records = gp.parse_all()

    if args.verbose:
        gp.print_summary()

    # 保存紧凑格式
    gp.save_compact_format(args.output, merge=args.merge)

    print(f"\n✅ 完成!")


if __name__ == "__main__":
    main()
