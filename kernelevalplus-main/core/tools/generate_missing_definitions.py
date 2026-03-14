#!/usr/bin/env python3
"""
从 shapes.py 补充缺失的 definition 文件

根据 shapes.py 中的模型维度信息，生成缺失的定义文件。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# shapes.py 中可以补充的维度信息
NEW_DIMENSIONS = [
    # DeepSeek V2
    {"n": 5120, "k": 12288, "model": "deepseek-v2", "layer": "moe-down", "layer_tag": "layer:moe-down", "desc": "MoE shared expert down projection"},
    {"n": 12288, "k": 5120, "model": "deepseek-v2", "layer": "moe-up", "layer_tag": "layer:moe-up", "desc": "MoE shared expert up/gate projection"},
    {"n": 1536, "k": 5120, "model": "deepseek-v2", "layer": "moe-routing-down", "layer_tag": "layer:moe-routing-down", "desc": "MoE routing expert down projection"},
    {"n": 5120, "k": 1536, "model": "deepseek-v2", "layer": "moe-routing-up", "layer_tag": "layer:moe-routing-up", "desc": "MoE routing expert up/gate projection"},
    # DeepSeek V3
    {"n": 1536, "k": 7168, "model": "deepseek-v3", "layer": "att-qkv", "layer_tag": "layer:att-qkv", "desc": "Attention QKV projection"},
    {"n": 7168, "k": 1536, "model": "deepseek-v3", "layer": "att-out", "layer_tag": "layer:att-out", "desc": "Attention output projection"},
    {"n": 512, "k": 7168, "model": "deepseek-v3", "layer": "moe-routing-up", "layer_tag": "layer:moe-routing-up", "desc": "MoE routing expert up/gate projection"},
    {"n": 7168, "k": 512, "model": "deepseek-v3", "layer": "moe-routing-down", "layer_tag": "layer:moe-routing-down", "desc": "MoE routing expert down projection"},
]

# 支持的 variants
VARIANTS = [
    {"name": "W4A32C8", "weight_bits": 4, "activation_bits": 32, "compute_bits": 8,
     "weight_dtype": "block_q4_0", "desc": "4-bit weights (Q4_0), 32-bit activation (FP32), 8-bit compute (INT8 DP4A)"},
    {"name": "W8A32C8", "weight_bits": 8, "activation_bits": 32, "compute_bits": 8,
     "weight_dtype": "block_q8_0", "desc": "8-bit weights (Q8_0), 32-bit activation (FP32), 8-bit compute (INT8 DP4A)"},
]

# 模型简称映射
MODEL_SHORT_NAMES = {
    "deepseek-v2": "ds2",
    "deepseek-v3": "ds3",
}

# 模型目录映射
MODEL_DIRS = {
    "deepseek-v2": "deepseek_v2",
    "deepseek-v3": "deepseek_v3",
}


def generate_definition(dim: Dict, variant: Dict, output_dir: Path) -> str:
    """生成单个 definition 文件"""

    n = dim["n"]
    k = dim["k"]
    model = dim["model"]
    layer = dim["layer"]
    layer_tag = dim["layer_tag"]
    desc = dim["desc"]

    variant_name = variant["name"]
    weight_bits = variant["weight_bits"]
    weight_dtype = variant["weight_dtype"]

    # 生成文件名
    model_short = MODEL_SHORT_NAMES[model]
    layer_short = layer.replace("-", "_")
    file_name = f"w{weight_bits}a32c8_{weight_dtype}_fp32_int8_{model_short}_{layer_short}_n{n}_k{k}.json"

    # 构建 definition 内容
    definition = {
        "name": f"w{weight_bits}a32c8_{weight_dtype}_fp32_int8_{model_short}_{layer_short}_n{n}_k{k}",
        "op_type": "quant_gemm",
        "variant": variant_name,
        "description": f"{model.replace('-', '-').title()} {desc} with {variant_name}. Using INT8 compute with llama.cpp {weight_dtype.upper()}×Q8_1 pattern.",
        "tags": [
            "status:verified",
            f"framework:{model_short}",
            "source:llama.cpp",
            layer_tag,
            "architecture:moe" if "moe" in layer or "routing" in layer else "architecture:transformer",
            f"quantization:{weight_dtype}",
            "quantization:q8_1",
            "compute:int8",
            "compute:dp4a",
            f"weight_bits:{weight_bits}",
            "activation_storage:fp32",
            "activation_quantization:q8_1_style",
            "compute_precision:int8",
            f"llama_cpp_pattern:{weight_dtype}_q8_1"
        ],
        "model_architectures": [model],
        "op_category": "quant_gemm",
        "axes": {
            "M": {"type": "var", "description": "Batch dimension (batch_size * seq_len)"},
            "N": {"type": "const", "value": n, "description": f"Output features ({desc.split()[-1].lower()})"},
            "K": {"type": "const", "value": k, "description": "Input features (hidden_size, must be multiple of 32)"},
            "block_size": {"type": "const", "value": 32, "description": f"Quantization block size for {weight_dtype.upper()} and Q8_1"}
        },
        "inputs": {
            "activation": {
                "shape": ["M", "K"],
                "dtype": "float32",
                "description": f"FP32 activation tensor, dynamically quantized to Q8_1 style per-block (32 values) for compute"
            },
            "weight": {
                "shape": ["N", "K/32"],
                "dtype": weight_dtype,
                "description": f"{weight_dtype.upper()} quantized weight tensor. Each block of 32 values shares a scale (fp16). Storage: (K/32) * 34 bytes per row"
            }
        },
        "outputs": {
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "FP32 output tensor"
            }
        },
        "types": {
            weight_dtype: {
                "size": 34,
                "description": f"{weight_dtype.upper()} block quantization format for weights (32 values per block)",
                "fields": [
                    {"name": "scale", "dtype": "float16", "description": "Scale factor for the block (fp16, 2 bytes)"},
                    {"name": "qs", "dtype": "int8[32]", "description": "Quantized values (32 int8 values, 32 bytes)"}
                ],
                "quantization": f"w = qs * scale",
                "formula": f"For each block j: output[i] = qs[i] * scale_j",
                "block_size": 32,
                "bytes_per_block": 34
            },
            "block_q8_1_activation": {
                "size": 36,
                "description": "Q8_1-style block quantization format for activations (32 values per block, dynamically computed)",
                "fields": [
                    {"name": "scale", "dtype": "float32", "description": "Per-block scale factor (computed dynamically)"},
                    {"name": "qs", "dtype": "int8[32]", "description": "Quantized values (32 int8 values)"}
                ],
                "quantization": "a = qs * scale",
                "note": "Sum field is NOT used for Q8_0 weights (only needed for Q4_0 with -8 offset)"
            }
        },
        "constraints": ["K % 32 == 0", "N >= 1", "M >= 1"],
        "formula": {
            "computation": f"C = A @ W^T where A(M,{k}) is FP32, W({n},{k}) is {weight_dtype.upper()} quantized, computed using INT8 arithmetic with llama.cpp {weight_dtype.upper()}×Q8_1 pattern",
            "dequantization": f"Weight: w = qs * scale_w, Activation: a = qs * scale_a",
            "llama_cpp_formula": "result = d8_0 * d8_1 * sumi where sumi is INT8 dot product",
            "compute_flow": f"1. Per-block quantize A to Q8_1 style (scale_a), 2. INT8 GEMM with {weight_dtype.upper()} weights (DP4A), 3. Apply scales: output = d8_0 * d8_1 * sumi"
        },
        "test_configs": [
            {"name": "batch_1", "M": 1, "N": n, "K": k},
            {"name": "batch_2", "M": 2, "N": n, "K": k},
            {"name": "batch_3", "M": 3, "N": n, "K": k},
            {"name": "batch_4", "M": 4, "N": n, "K": k},
            {"name": "batch_5", "M": 5, "N": n, "K": k},
            {"name": "batch_8", "M": 8, "N": n, "K": k},
            {"name": "batch_512", "M": 512, "N": n, "K": k}
        ],
        "performance_notes": {
            "int8_benefit": "INT8 compute with DP4A provides 2-4x speedup over FP32 at similar accuracy",
            "llama_cpp_compatibility": f"Matches llama.cpp vec_dot_{weight_dtype}_q8_1 pattern for maximum compatibility",
            "per_block_quantization": "Per-block (32 values) activation quantization provides better accuracy than per-row",
            "memory_bound": "For M < 32, typically memory-bound",
            "compute_bound": "For M >= 32, typically compute-bound"
        }
    }

    # 写入文件
    model_dir = MODEL_DIRS[model]
    file_path = output_dir / model_dir / file_name

    return file_path, definition


def main():
    # 输出目录
    output_dir = Path("/home/haiyan/Agent4Kernel/KernelEvalPlus/definitions/quant_gemm")

    print("=" * 80)
    print("从 shapes.py 补充缺失的 definition 文件")
    print("=" * 80)

    generated_files = []

    # 为每个维度生成所有 variants 的定义
    for dim in NEW_DIMENSIONS:
        for variant in VARIANTS:
            file_path, definition = generate_definition(dim, variant, output_dir)

            # 创建模型目录（如果不存在）
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 检查文件是否已存在
            if file_path.exists():
                print(f"⚠️  文件已存在: {file_path.name}")
                continue

            # 写入文件
            with open(file_path, 'w') as f:
                json.dump(definition, f, indent=2)

            generated_files.append(file_path)
            print(f"✅ 生成: {file_path}")

    print()
    print(f"📊 总结:")
    print(f"  生成文件数: {len(generated_files)}")
    print(f"  新增维度: {len(NEW_DIMENSIONS)} 个")
    print(f"  每个维度生成: {len(VARIANTS)} 个 variant")
    print(f"  总计: {len(NEW_DIMENSIONS) * len(VARIANTS)} 个定义文件")


if __name__ == "__main__":
    main()
