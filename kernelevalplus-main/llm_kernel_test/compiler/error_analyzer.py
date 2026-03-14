#!/usr/bin/env python3
"""
编译错误分析模块

提供 CUDA 编译错误的诊断和修复建议。
"""

import re
from pathlib import Path
from typing import List, Dict, Optional


def analyze_compilation_error(error_msg: str, kernel_file: Path) -> List[Dict]:
    """
    分析编译错误，提供具体的修复建议

    Args:
        error_msg: 编译错误消息
        kernel_file: kernel 源文件路径（用于读取代码分析）

    Returns:
        诊断信息列表，每个包含:
        - type: 错误类型标识
        - problem: 问题描述
        - suggestion: 修复建议
    """
    diagnostics = []
    error_lower = error_msg.lower()

    # 读取 kernel 源码用于诊断
    kernel_code = ""
    try:
        with open(kernel_file, 'r') as f:
            kernel_code = f.read()
    except:
        pass

    # 错误模式 1: FP16 转换错误 (最常见，优先检测)
    if ('no suitable constructor' in error_lower and '__half' in error_msg) or \
       '__half2float' in error_msg and ('uint16_t' in error_msg or 'w_block->d' in error_msg):
        diagnostics.append({
            "type": "fp16_conversion",
            "problem": "直接使用 __half2float(block->d) 转换 uint16_t 导致编译错误",
            "suggestion": "使用 union 方式转换:\n"
                         "     __device__ float read_half_as_float(uint16_t h) {\n"
                         "         union { uint16_t u16; __half f16; } un;\n"
                         "         un.u16 = h;\n"
                         "         return __half2float(un.f16);\n"
                         "     }\n"
                         "     然后使用: float d_w = read_half_as_float(w_block->d);"
        })

    # 错误模式 2: 缺少 torch/extension.h
    if 'torch/extension.h' in error_msg or \
       ('torch' in error_lower and 'no such file' in error_lower) or \
       ('pybind11' in error_lower and 'not declared' in error_lower):
        diagnostics.append({
            "type": "missing_torch_header",
            "problem": "缺少 PyTorch 扩展头文件",
            "suggestion": "在 kernel.cu 开头添加:\n"
                         "     #include <torch/extension.h>"
        })

    # 错误模式 3: 缺少 PYBIND11_MODULE
    if 'torch_extension_name' in error_lower or \
       ('pybind11_module' in error_lower and 'not declared' in error_lower):
        if 'PYBIND11_MODULE' not in kernel_code:
            diagnostics.append({
                "type": "missing_pybind_module",
                "problem": "缺少 PYBIND11_MODULE 绑定",
                "suggestion": "在 kernel.cu 末尾添加:\n"
                             "     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
                             "         m.def(\"forward\", &forward, \"Kernel forward\");\n"
                             "     }"
            })

    # 错误模式 4: 结构体未定义
    missing_structs = []
    if 'block_q8_0' in error_msg and ('undefined' in error_lower or 'undeclared' in error_lower):
        missing_structs.append('block_q8_0')
    if 'block_q8_1' in error_msg and ('undefined' in error_lower or 'undeclared' in error_lower):
        missing_structs.append('block_q8_1')
    if 'block_q4_0' in error_msg and ('undefined' in error_lower or 'undeclared' in error_lower):
        missing_structs.append('block_q4_0')

    if missing_structs:
        struct_defs = {
            'block_q8_0': "typedef struct {\n         uint16_t d;        // FP16 scale\n         int8_t qs[32];     // quantized values\n     } block_q8_0;\n     static_assert(sizeof(block_q8_0) == 34, \"\");",
            'block_q8_1': "typedef struct {\n         uint32_t ds;       // d and s packed as half2\n         int8_t qs[32];     // quantized values\n     } block_q8_1;\n     static_assert(sizeof(block_q8_1) == 36, \"\");",
            'block_q4_0': "typedef struct {\n         uint16_t d;        // FP16 scale\n         uint8_t qs[16];    // packed 4-bit values\n     } block_q4_0;\n     static_assert(sizeof(block_q4_0) == 18, \"\");"
        }
        suggestion = "添加以下结构体定义:\n"
        for s in missing_structs:
            suggestion += f"\n     // {s}\n     {struct_defs.get(s, '')}\n"

        diagnostics.append({
            "type": "missing_struct",
            "problem": f"结构体未定义: {', '.join(missing_structs)}",
            "suggestion": suggestion
        })

    # 错误模式 5: extern "C" 与 PyTorch 不兼容
    if 'extern "C"' in kernel_code and 'PYBIND11_MODULE' not in kernel_code:
        diagnostics.append({
            "type": "extern_c_style",
            "problem": "使用了 extern \"C\" 风格，但测试框架需要 PyTorch 绑定",
            "suggestion": "将 extern \"C\" 函数改为 PyTorch 绑定:\n"
                         "     torch::Tensor forward(torch::Tensor weight, ...) {\n"
                         "         // 实现\n"
                         "     }\n"
                         "     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
                         "         m.def(\"forward\", &forward);\n"
                         "     }"
        })

    # 错误模式 6: 占位符注释
    if '/* extract' in kernel_code or '// TODO' in kernel_code:
        if '/* extract' in kernel_code:
            diagnostics.append({
                "type": "placeholder_code",
                "problem": "代码包含未实现的占位符注释",
                "suggestion": "将占位符替换为实际实现代码"
            })

    # 错误模式 7: 缺少必要的头文件
    missing_headers = []
    if 'cuda_runtime.h' not in kernel_code and 'cudaError' in error_msg:
        missing_headers.append('cuda_runtime.h')
    if 'cuda_fp16.h' not in kernel_code and ('__half' in error_msg or '__half2float' in error_msg):
        missing_headers.append('cuda_fp16.h')
    if 'stdint.h' not in kernel_code and ('uint16_t' in error_msg or 'int8_t' in error_msg):
        missing_headers.append('stdint.h')

    if missing_headers:
        diagnostics.append({
            "type": "missing_headers",
            "problem": f"缺少必要的头文件: {', '.join(missing_headers)}",
            "suggestion": "添加以下 #include:\n" +
                         '\n'.join(f"     #include <{h}>" for h in missing_headers)
        })

    # 错误模式 8: __half2float 未定义 (缺少 cuda_fp16.h 或 CUDA 版本问题)
    if '__half2float' in error_msg and 'undefined' in error_lower:
        if 'cuda_fp16.h' in kernel_code:
            # 有头文件但仍然未定义，可能是宏问题
            diagnostics.append({
                "type": "half2float_undefined",
                "problem": "__half2float 未定义，可能是编译标志问题",
                "suggestion": "PyTorch 默认禁用 half 操作。使用 union 转换替代:\n"
                             "     __device__ float read_half_as_float(uint16_t h) {\n"
                             "         union { uint16_t u16; __half f16; } un;\n"
                             "         un.u16 = h;\n"
                             "         return __half2float(un.f16);\n"
                             "     }"
            })
        else:
            diagnostics.append({
                "type": "half2float_undefined",
                "problem": "__half2float 未定义，缺少头文件",
                "suggestion": "添加 #include <cuda_fp16.h>"
            })

    return diagnostics


class ErrorAnalyzer:
    """编译错误分析器"""

    def __init__(self):
        pass

    def analyze(self, error_msg: str, kernel_file: Path) -> List[Dict]:
        """
        分析编译错误

        Args:
            error_msg: 编译错误消息
            kernel_file: kernel 源文件路径

        Returns:
            诊断信息列表
        """
        return analyze_compilation_error(error_msg, kernel_file)

    def print_diagnostics(self, diagnostics: List[Dict]) -> None:
        """打印诊断信息到控制台"""
        if not diagnostics:
            return

        print(f"\n  🔍 编译错误诊断:")
        for diag in diagnostics:
            print(f"     问题: {diag['problem']}")
            print(f"     建议: {diag['suggestion']}")
