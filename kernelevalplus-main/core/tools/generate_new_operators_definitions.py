#!/usr/bin/env python3
"""
从 baseline 数据生成完整的新算子 definition 文件

覆盖所有 baseline 数据：
- RMS Norm: 8 个 hidden_size
- TopK: 2 个 k 值 × 2 个 ne0 值
- Flash Attention: 2 个模型 × 3 个 KV 类型 × 3 个 cache size
"""
import json
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
BASELINE_DIR = PROJECT_ROOT / "data" / "baseline"
DEFINITIONS_DIR = PROJECT_ROOT / "definitions"


def load_baseline(filename):
    """加载 baseline 数据"""
    with open(BASELINE_DIR / filename) as f:
        return json.load(f)


def save_definition(op_type, model, filename, definition):
    """保存 definition 文件"""
    dir_path = DEFINITIONS_DIR / op_type / model
    dir_path.mkdir(parents=True, exist_ok=True)

    with open(dir_path / filename, 'w') as f:
        json.dump(definition, f, indent=2)

    print(f"  Created: {op_type}/{model}/{filename}")


# ==================== RMS Norm ====================

def generate_rms_norm_definitions():
    """生成 RMS Norm definition 文件"""
    print("\n[RMS Norm]")

    # 从 baseline 获取所有 hidden_size
    rms_data = load_baseline("rms_norm_baseline.json")
    hidden_sizes = sorted(set(d["hidden_size"] for d in rms_data.values()))

    # 模型映射：hidden_size -> 模型名称
    model_mapping = {
        128: ("llama", "llama2_7b"),      # LLaMA2-7B
        512: ("llama", "llama2_7b"),      # LLaMA2-7B (small variant)
        1536: ("qwen", "qwen3_4b"),       # Qwen3-4B
        2560: ("qwen", "qwen3_4b"),       # Qwen3-4B variant
        3584: ("qwen", "qwen2_5_7b"),     # Qwen2.5-7B
        4096: ("llama", "llama3_8b"),     # LLaMA-3-8B
        5120: ("qwen", "qwen2_5_7b"),     # Qwen2.5-7B variant
        7168: ("deepseek", "deepseek_v3"),  # DeepSeek-V3
    }

    created = 0
    for hs in hidden_sizes:
        if hs not in model_mapping:
            print(f"  Skipped: hs={hs} (no model mapping)")
            continue

        model_dir, model_name = model_mapping[hs]

        definition = {
            "name": f"fp32_rms_norm_{model_name}_hs{hs}",
            "op_type": "rms_norm",
            "op_category": "normalization",
            "variant": "FP32",
            "description": f"RMS Norm for {model_name} with hidden_size={hs}",
            "tags": [
                "status:generated",
                "framework:ggml",
                "source:ggml-python",
                "layer:norm",
                "normalization:rms",
                "activation:fp32",
                f"model:{model_name}"
            ],
            "model_architectures": [model_name],
            "axes": {
                "batch_size": {"type": "var", "description": "Batch/sequence dimension"},
                "hidden_size": {"type": "const", "value": hs, "description": "Feature dimension"},
                "epsilon": {"type": "const", "value": 1e-6, "description": "Numerical stability constant"}
            },
            "inputs": {
                "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32", "description": "Input tensor"},
                "weight": {"shape": ["hidden_size"], "dtype": "float32", "description": "Learned scale parameter"}
            },
            "outputs": {
                "output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32", "description": "Normalized output"}
            },
            "formula": {
                "computation": "output = input / sqrt(mean(input^2, axis=-1) + epsilon) * weight"
            },
            "test_configs": [
                {"name": "batch_1", "batch_size": 1},
                {"name": "batch_8", "batch_size": 8},
                {"name": "batch_512", "batch_size": 512}
            ],
            "baseline_ref": {
                "case_id": f"rms_norm_hs{hs}_{hs}x512x1x1",
                "source": "ggml-python",
                "metric": "gbps"
            }
        }

        filename = f"fp32_rms_norm_{model_name}_hs{hs}.json"
        save_definition("rms_norm", model_dir, filename, definition)
        created += 1

    print(f"  Total: {created} definitions")


# ==================== TopK ====================

def generate_topk_definitions():
    """生成 TopK definition 文件"""
    print("\n[TopK]")

    # 配置：k 值 × ne0 值
    configs = [
        (6, 160, "llama", "llama3_8b"),
        (6, 256, "llama", "llama3_8b"),
        (8, 160, "llama", "llama3_8b"),
        (8, 256, "llama", "llama3_8b"),
        (6, 160, "qwen", "qwen2_5_7b"),
        (6, 256, "qwen", "qwen2_5_7b"),
        (8, 160, "qwen", "qwen2_5_7b"),
        (8, 256, "qwen", "qwen2_5_7b"),
    ]

    created = 0
    for k, ne0, model_dir, model_name in configs:
        definition = {
            "name": f"fp32_top_k_sampling_{model_name}_k{k}_ne0{ne0}",
            "op_type": "topk",
            "op_category": "sampling",
            "variant": "FP32",
            "description": f"Top-{k} sampling for {model_name} with vocab_subset={ne0}",
            "tags": [
                "status:generated",
                "framework:ggml",
                "source:ggml-python",
                "layer:sampling",
                "sampling:top_k",
                "activation:fp32",
                f"model:{model_name}"
            ],
            "model_architectures": [model_name],
            "axes": {
                "batch_size": {"type": "var", "description": "Batch dimension"},
                "vocab_subset": {"type": "const", "value": ne0, "description": "Vocabulary subset size for testing"},
                "k": {"type": "const", "value": k, "description": "Top-K value"}
            },
            "inputs": {
                "probs": {"shape": ["batch_size", "vocab_subset"], "dtype": "float32", "description": "Probability distribution"}
            },
            "outputs": {
                "samples": {"shape": ["batch_size"], "dtype": "int64", "description": "Sampled token indices"}
            },
            "formula": {
                "computation": f"top_indices = argsort(probs, descending=True)[:{k}]; samples = categorical(probs[top_indices])"
            },
            "test_configs": [
                {"name": "batch_1", "batch_size": 1},
                {"name": "batch_8", "batch_size": 8},
                {"name": "batch_512", "batch_size": 512}
            ],
            "baseline_ref": {
                "case_id": f"topk_k{k}_ne0{ne0}_{ne0}x512x1x1",
                "source": "ggml-python",
                "metric": "gbps"
            }
        }

        filename = f"fp32_top_k_sampling_{model_name}_k{k}_ne0{ne0}.json"
        save_definition("topk", model_dir, filename, definition)
        created += 1

    print(f"  Total: {created} definitions")


# ==================== Flash Attention ====================

def generate_flash_attention_definitions():
    """生成 Flash Attention definition 文件"""
    print("\n[Flash Attention]")

    # 配置：模型 × KV类型 × cache_size
    configs = []
    models = [
        ("llama", "Llama3-8B", "llama3_8b", 32, 128),      # LLaMA-3-8B
        ("qwen", "Qwen2.5-7B", "qwen2_5_7b", 28, 128),    # Qwen2.5-7B
    ]
    kv_types = ["F16", "Q4_0", "Q8_0"]
    cache_sizes = [512, 4096, 8192]

    for model_dir, model_display, model_name, num_heads, head_dim in models:
        for kv_type in kv_types:
            for cache_size in cache_sizes:
                configs.append((model_dir, model_display, model_name, kv_type, cache_size, num_heads, head_dim))

    created = 0
    for model_dir, model_display, model_name, kv_type, cache_size, num_heads, head_dim in configs:
        kv_type_lower = kv_type.lower()

        definition = {
            "name": f"fp32_flash_attention_{model_name}_{kv_type_lower}_cache{cache_size}",
            "op_type": "flash_attention",
            "op_category": "attention",
            "variant": f"FP32_{kv_type}",
            "description": f"Flash Attention for {model_display} with KV cache size {cache_size} and {kv_type} KV storage",
            "tags": [
                "status:generated",
                "framework:ggml",
                "source:ggml-python",
                "layer:attention",
                "attention:flash",
                f"kv_cache:{kv_type_lower}",
                "activation:fp32",
                f"model:{model_name}"
            ],
            "model_architectures": [model_name],
            "axes": {
                "batch_size": {"type": "var", "description": "Number of attention blocks (nb)"},
                "seq_len": {"type": "const", "value": cache_size, "description": "KV cache sequence length"},
                "num_heads": {"type": "const", "value": num_heads, "description": "Number of attention heads"},
                "head_dim": {"type": "const", "value": head_dim, "description": "Dimension per head"}
            },
            "inputs": {
                "query": {"shape": ["batch_size", "num_heads", "head_dim"], "dtype": "float32", "description": "Query projection"},
                "key_cache": {"shape": ["seq_len", "num_heads", "head_dim"], "dtype": "float16" if kv_type == "F16" else "int8", "description": "Cached key projections"},
                "value_cache": {"shape": ["seq_len", "num_heads", "head_dim"], "dtype": "float16" if kv_type == "F16" else "int8", "description": "Cached value projections"}
            },
            "outputs": {
                "output": {"shape": ["batch_size", "num_heads", "head_dim"], "dtype": "float32", "description": "Attention output"}
            },
            "formula": {
                "computation": "attn = softmax(Q @ K^T / sqrt(d)) @ V"
            },
            "test_configs": [
                {"name": "batch_1", "batch_size": 1},
                {"name": "batch_8", "batch_size": 8},
                {"name": "batch_512", "batch_size": 512}
            ],
            "baseline_ref": {
                "case_id": f"flash_attn_{model_display}_{kv_type}_cache{cache_size}_nb512",
                "source": "ggml-python",
                "metric": "tflops"
            }
        }

        filename = f"fp32_flash_attention_{model_name}_{kv_type_lower}_cache{cache_size}.json"
        save_definition("flash_attention", model_dir, filename, definition)
        created += 1

    print(f"  Total: {created} definitions")


def main():
    print("=" * 60)
    print("生成新算子 Definition 文件")
    print("=" * 60)

    generate_rms_norm_definitions()
    generate_topk_definitions()
    generate_flash_attention_definitions()

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
