#!/usr/bin/env python3
"""
Reference 模块

提供量化函数和 GEMM 参考实现，用于验证 CUDA kernel 的正确性。

模块结构:
    quantize.py - 量化函数 (Q4_0, Q4_1, Q8_0, Q8_1)
    gemm_ref.py - GEMM 参考实现
"""

from .quantize import (
    quantize_to_q8_0,
    quantize_to_q4_0,
    quantize_to_q4_1,
    quantize_to_q8_1,
)

from .gemm_ref import (
    reference_q8_0_gemm,
    reference_q4_0_fp32_gemm,
    reference_q4_0_q8_1_gemm,
    reference_q4_1_fp32_gemm,
    reference_q4_1_q8_1_gemm,
)

__all__ = [
    # 量化函数
    "quantize_to_q8_0",
    "quantize_to_q4_0",
    "quantize_to_q4_1",
    "quantize_to_q8_1",
    # GEMM 参考实现
    "reference_q8_0_gemm",
    "reference_q4_0_fp32_gemm",
    "reference_q4_0_q8_1_gemm",
    "reference_q4_1_fp32_gemm",
    "reference_q4_1_q8_1_gemm",
]
