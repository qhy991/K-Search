#!/usr/bin/env python3
"""
从 GGML 基线数据生成 Definition JSON 文件

根据 baseline_data_compact.json 中的维度配置，自动生成对应的 definition JSON 文件。

Usage:
    # 生成所有可用基线对应的 definitions
    python -m python.tools.generate_definitions_from_baseline

    # 只生成特定类型的
    python -m python.tools.generate_definitions_from_baseline --quant-type q4_0

    # 指定输出目录
    python -m python.tools.generate_definitions_from_baseline --output-dir definitions/quant_gemm/generated
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ggml_baseline_api import BaselineAPI


# 模型配置映射 (根据 N, K 推断可能的模型层)
MODEL_LAYERS_CONFIG = {
    # Llama 3 8B
    (4096, 4096): {
        "model": "llama3_8b",
        "layer": "att_out",
        "description": "LLaMA-3-8B Attention Output projection"
    },
    (14336, 4096): {
        "model": "llama3_8b",
        "layer": "ffn_up",
        "description": "LLaMA-3-8B FFN Up/Gate projection"
    },
    (4096, 14336): {
        "model": "llama3_8b",
        "layer": "ffn_down",
        "description": "LLaMA-3-8B FFN Down projection"
    },
    (128256, 4096): {
        "model": "llama3_8b",
        "layer": "lm_head",
        "description": "LLaMA-3-8B LM Head projection"
    },

    # Llama 3 70B
    (8192, 8192): {
        "model": "llama3_70b",
        "layer": "att_out",
        "description": "LLaMA-3-70B Attention Output projection"
    },

    # Qwen 2.5 7B
    (3584, 3584): {
        "model": "qwen2_5_7b",
        "layer": "att_out",
        "description": "Qwen-2.5-7B Attention Output projection"
    },
    (18944, 3584): {
        "model": "qwen2_5_7b",
        "layer": "ffn_up",
        "description": "Qwen-2.5-7B FFN Up/Gate projection"
    },
    (3584, 18944): {
        "model": "qwen2_5_7b",
        "layer": "ffn_down",
        "description": "Qwen-2.5-7B FFN Down projection"
    },
    (152064, 3584): {
        "model": "qwen2_5_7b",
        "layer": "lm_head",
        "description": "Qwen-2.5-7B LM Head projection"
    },

    # Qwen 3 系列
    (3072, 1024): {
        "model": "qwen3_0_6b",
        "layer": "att_out",
        "description": "Qwen3-0.6B Attention Output projection"
    },
    (7680, 2560): {
        "model": "qwen3_1_5b",
        "layer": "att_out",
        "description": "Qwen3-1.5B Attention Output projection"
    },
    (2560, 2560): {
        "model": "qwen3_4b",
        "layer": "att_out",
        "description": "Qwen3-4B Attention Output projection"
    },
    (9728, 2560): {
        "model": "qwen3_4b",
        "layer": "ffn_up",
        "description": "Qwen3-4B FFN Up/Gate projection"
    },
    (2560, 9728): {
        "model": "qwen3_4b",
        "layer": "ffn_down",
        "description": "Qwen3-4B FFN Down projection"
    },
    (151936, 2560): {
        "model": "qwen3_4b",
        "layer": "lm_head",
        "description": "Qwen3-4B LM Head projection"
    },
    (12288, 4096): {
        "model": "qwen3_8b",
        "layer": "att_qkv",
        "description": "Qwen3-8B Attention QKV projection"
    },
    (17408, 5120): {
        "model": "qwen3_14b",
        "layer": "ffn_up",
        "description": "Qwen3-14B FFN Up/Gate projection"
    },
    (25600, 5120): {
        "model": "qwen3_32b",
        "layer": "ffn_up",
        "description": "Qwen3-32B FFN Up/Gate projection"
    },

    # DeepSeek V2
    (5120, 5120): {
        "model": "deepseek_v2",
        "layer": "att_out",
        "description": "DeepSeek-V2 Attention Output projection"
    },
    (102400, 5120): {
        "model": "deepseek_v2",
        "layer": "lm_head",
        "description": "DeepSeek-V2 LM Head projection"
    },

    # DeepSeek V3
    (7168, 7168): {
        "model": "deepseek_v3",
        "layer": "att_out",
        "description": "DeepSeek-V3 Attention Output projection"
    },
    (21504, 7168): {
        "model": "deepseek_v3",
        "layer": "att_qkv",
        "description": "DeepSeek-V3 Attention QKV projection"
    },
    (18432, 7168): {
        "model": "deepseek_v3",
        "layer": "moe_up",
        "description": "DeepSeek-V3 MoE Up/Gate projection"
    },
    (7168, 18432): {
        "model": "deepseek_v3",
        "layer": "moe_down",
        "description": "DeepSeek-V3 MoE Down projection"
    },
    (129280, 7168): {
        "model": "deepseek_v3",
        "layer": "lm_head",
        "description": "DeepSeek-V3 LM Head projection"
    },

    # Mistral
    (14336, 4096): {
        "model": "mistral7b",
        "layer": "ffn_up",
        "description": "Mistral-7B FFN Up/Gate projection"
    },

    # Mixtral
    (14336, 4096): {
        "model": "mixtral8x7b",
        "layer": "moe_up",
        "description": "Mixtral-8x7B MoE Up/Gate projection"
    },
}


def get_variant_info(type_a: str) -> Tuple[str, str, str]:
    """根据量化类型获取 variant 信息

    Returns:
        (variant_name, weight_bits, compute_bits)
    """
    variant_map = {
        "q4_0": ("W4A32C8", "4", "8"),
        "q4_1": ("W4A32C8", "4", "8"),
        "q8_0": ("W8A32C8", "8", "8"),
    }
    return variant_map.get(type_a, ("W4A32C8", "4", "8"))


def get_weight_block_type(type_a: str) -> str:
    """获取权重 block 类型"""
    block_type_map = {
        "q4_0": "block_q4_0",
        "q4_1": "block_q4_1",
        "q8_0": "block_q8_0",
    }
    return block_type_map.get(type_a, "block_q4_0")


def get_formula(type_a: str) -> Dict:
    """获取计算公式"""
    formula_map = {
        "q4_0": {
            "computation": "C = A @ W^T where A(M,K) is FP32, W(N,K/32) is Q4_0 quantized",
            "llama_cpp_formula": "result = d4_0 * (d8_1 * sumi - 8 * s8_1)",
            "explanation": "Q4_0 uses offset-8 encoding. The -8*s8_1 term compensates for this offset.",
        },
        "q4_1": {
            "computation": "C = A @ W^T where A(M,K) is FP32, W(N,K/32) is Q4_1 quantized",
            "llama_cpp_formula": "result = d4_1 * d8_1 * sumi + m4_1 * s8_1 / 4",
            "explanation": "Q4_1 uses asymmetric quantization with min value.",
        },
        "q8_0": {
            "computation": "C = A @ W^T where A(M,K) is FP32, W(N,K/32) is Q8_0 quantized",
            "llama_cpp_formula": "result = d8_0 * d8_1 * sumi",
            "explanation": "Q8_0 × Q8_1 pattern with symmetric quantization.",
        },
    }
    return formula_map.get(type_a, formula_map["q4_0"])


def generate_definition(
    case_id: str,
    case_data: Dict,
    config: Dict,
    output_dir: Path
) -> str:
    """生成单个 definition JSON 文件

    Args:
        case_id: Baseline case_id (例如: w4a32c8_q4_0_f32_m4096_n1_k4096)
        case_data: Case 数据
        config: 模型层配置
        output_dir: 输出目录

    Returns:
        生成的文件路径
    """
    type_a = case_data["type_a"]
    n = case_data["m"]  # GGML 的 M 对应我们的 N
    k = case_data["k"]
    m = case_data["n"]  # GGML 的 N 对应我们的 M

    variant, weight_bits, compute_bits = get_variant_info(type_a)
    weight_block_type = get_weight_block_type(type_a)
    formula = get_formula(type_a)

    model = config["model"]
    layer = config["layer"]
    description = config["description"]

    # 构建文件名
    filename = f"w{weight_bits}a32c{compute_bits}_{type_a}_q8_1_{model}_{layer}_n{n}_k{k}.json"

    # 构建 definition
    definition = {
        "name": filename,
        "op_type": "quant_gemm",
        "variant": variant,
        "description": f"{description} with {variant}. Generated from GGML baseline.",
        "tags": [
            "status:generated",
            "framework:ggml",
            "source:ggml-python",
            f"layer:{layer.replace('_', '-')}",
            f"quantization:{type_a}",
            "quantization:q8_1",
            "compute:int8",
            "compute:dp4a",
            f"weight_bits:{weight_bits}",
            "activation_storage:fp32",
        ],
        "model_architectures": [model.replace("_", "-")],
        "op_category": "quant_gemm",
        "axes": {
            "M": {"type": "var", "description": "Batch dimension"},
            "N": {"type": "const", "value": n, "description": "Output features"},
            "K": {"type": "const", "value": k, "description": "Input features"},
            "block_size": {"type": "const", "value": 32, "description": "Quantization block size"},
        },
        "inputs": {
            "activation": {
                "shape": ["M", "K"],
                "dtype": "float32",
                "description": "FP32 activation tensor, dynamically quantized to Q8_1 per-block during compute"
            },
            "weight": {
                "shape": ["N", "K/32"],
                "dtype": weight_block_type,
                "description": f"{type_a.upper()} quantized weight tensor"
            }
        },
        "outputs": {
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "FP32 output tensor"
            }
        },
        "constraints": ["K % 32 == 0"],
        "formula": formula,
        "test_configs": [
            {"name": "single_token", "M": 1, "N": n, "K": k},
            {"name": "small_batch", "M": 4, "N": n, "K": k},
            {"name": "medium_batch", "M": 16, "N": n, "K": k},
        ],
        "baseline_ref": {
            "case_id": case_id,
            "source": "ggml-python",
            "note": "Generated from GGML baseline performance data"
        }
    }

    # 写入文件
    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = model_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(definition, f, indent=2, ensure_ascii=False)

    return str(file_path)


def generate_definitions_from_baseline(
    quant_type: str = None,
    output_dir: str = None,
    hardware: str = "RTX4090"
) -> Dict:
    """
    从基线数据生成 definition 文件

    Args:
        quant_type: 只生成特定量化类型 (None = 全部)
        output_dir: 输出目录
        hardware: 硬件名称 (用于筛选有数据的配置)

    Returns:
        统计信息字典
    """
    # 初始化 API
    api = BaselineAPI()

    # 设置输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "definitions" / "quant_gemm" / "generated"
    else:
        output_dir = Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "by_model": {},
        "files": []
    }

    print(f"\n{'='*70}")
    print(f"从 GGML 基线数据生成 Definition 文件")
    print(f"{'='*70}")
    print(f"\n输出目录: {output_dir}")
    print(f"硬件: {hardware}")
    if quant_type:
        print(f"量化类型: {quant_type}")

    # 遍历所有基线 case
    for case_id, case_data in api.data.items():
        type_a = case_data["type_a"]

        # 过滤量化类型
        if quant_type and type_a != quant_type:
            continue

        # 检查硬件是否有数据
        hw_data = case_data["hardware"].get(hardware)
        if not hw_data:
            stats["skipped"] += 1
            continue

        stats["total"] += 1

        # 获取维度
        m = case_data["m"]  # GGML M = output = our N
        k = case_data["k"]

        # 查找模型层配置
        config_key = (m, k)
        if config_key not in MODEL_LAYERS_CONFIG:
            # 尝试通用匹配
            stats["skipped"] += 1
            continue

        config = MODEL_LAYERS_CONFIG[config_key]

        # 生成 definition
        try:
            file_path = generate_definition(case_id, case_data, config, output_dir)
            stats["generated"] += 1
            stats["files"].append(file_path)

            # 统计模型
            model = config["model"]
            if model not in stats["by_model"]:
                stats["by_model"][model] = 0
            stats["by_model"][model] += 1

            print(f"  ✓ {file_path}")

        except Exception as e:
            print(f"  ✗ 生成失败: {case_id} - {e}")
            stats["skipped"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="从 GGML 基线数据生成 Definition JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 生成所有
  python -m python.tools.generate_definitions_from_baseline

  # 只生成 Q4_0
  python -m python.tools.generate_definitions_from_baseline --quant-type q4_0

  # 指定输出目录
  python -m python.tools.generate_definitions_from_baseline --output-dir definitions/quant_gemm/baseline
        """
    )

    parser.add_argument(
        "--quant-type",
        type=str,
        choices=["q4_0", "q4_1", "q8_0"],
        help="只生成特定量化类型"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="definitions/quant_gemm/generated",
        help="输出目录"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="RTX4090",
        help="硬件名称 (用于筛选)"
    )

    args = parser.parse_args()

    # 生成 definitions
    stats = generate_definitions_from_baseline(
        quant_type=args.quant_type,
        output_dir=args.output_dir,
        hardware=args.hardware
    )

    # 打印统计
    print(f"\n{'='*70}")
    print(f"生成完成")
    print(f"{'='*70}")
    print(f"总计: {stats['total']}")
    print(f"已生成: {stats['generated']}")
    print(f"跳过: {stats['skipped']}")

    if stats["by_model"]:
        print(f"\n按模型统计:")
        for model, count in sorted(stats["by_model"].items()):
            print(f"  {model}: {count}")


if __name__ == "__main__":
    main()
