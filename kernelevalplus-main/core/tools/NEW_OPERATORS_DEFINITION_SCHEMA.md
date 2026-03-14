# 新算子 Definition Schema 设计

## 一、FlashInfer-Bench 格式分析

### RMS Norm 示例 (FlashInfer-Bench)
```
Name: rmsnorm_h2048
Description: Root Mean Square Normalization with hidden_size=2048

Axes:
  batch_size: var
  hidden_size: 2048 (const)

Inputs:
  hidden_states: bfloat16 [batch_size, hidden_size]
  weight: bfloat16 [hidden_size]

Outputs:
  output: bfloat16 [batch_size, hidden_size]
```

### TopK Sampling 示例 (FlashInfer-Bench)
```
Name: top_k_sampling_from_probs_v128256
Description: Top-k sampling from probabilities with vocab_size=128256

Axes:
  batch_size: var
  vocab_size: 128256 (const)

Inputs:
  probs: float32 [batch_size, vocab_size]
  top_k: int32 [batch_size]

Outputs:
  samples: int64 [batch_size]
```

---

## 二、统一 Schema 设计

### 设计原则

1. **兼容现有 GEMM schema** - 保持与 quant_gemm 的一致性
2. **支持 FlashInfer 风格** - 简洁的 axes 和 signature 定义
3. **明确性能指标** - 区分 compute-bound (TFLOPS) 和 memory-bound (GB/s)

### 通用结构

```json
{
  "name": "算子名称",
  "op_type": "算子类型",
  "op_category": "算子分类",
  "variant": "变体标识",
  "description": "算子描述",
  "tags": ["标签列表"],
  "model_architectures": ["适用模型列表"],

  "axes": {
    "维度名": {"type": "const|var", "value": 数值, "description": "描述"}
  },

  "inputs": {
    "输入名": {"shape": ["维度"], "dtype": "类型", "description": "描述"}
  },

  "outputs": {
    "输出名": {"shape": ["维度"], "dtype": "类型", "description": "描述"}
  },

  "formula": {
    "computation": "计算公式",
    "explanation": "解释说明"
  },

  "test_configs": [
    {"name": "配置名", "维度值": ...}
  ],

  "baseline_ref": {
    "case_id": "基线ID",
    "source": "数据源",
    "metric": "tflops|gbps"
  }
}
```

---

## 三、RMS Norm Definition

### 完整示例

```json
{
  "name": "fp32_rms_norm_llama3_8b_hs4096",
  "op_type": "rms_norm",
  "op_category": "normalization",
  "variant": "FP32",
  "description": "Root Mean Square Normalization for LLaMA-3-8B with hidden_size=4096. Epsilon is fixed at 1e-6.",
  "tags": [
    "status:generated",
    "framework:ggml",
    "source:ggml-python",
    "layer:norm",
    "normalization:rms",
    "activation:fp32"
  ],
  "model_architectures": ["llama3-8b"],

  "axes": {
    "batch_size": {
      "type": "var",
      "description": "Batch/sequence dimension"
    },
    "hidden_size": {
      "type": "const",
      "value": 4096,
      "description": "Feature dimension"
    },
    "epsilon": {
      "type": "const",
      "value": 1e-6,
      "description": "Small constant for numerical stability"
    }
  },

  "inputs": {
    "hidden_states": {
      "shape": ["batch_size", "hidden_size"],
      "dtype": "float32",
      "description": "Input tensor to normalize"
    },
    "weight": {
      "shape": ["hidden_size"],
      "dtype": "float32",
      "description": "Learned scale/gamma parameter"
    }
  },

  "outputs": {
    "output": {
      "shape": ["batch_size", "hidden_size"],
      "dtype": "float32",
      "description": "Normalized output tensor"
    }
  },

  "formula": {
    "computation": "output = input / sqrt(mean(input^2) + epsilon) * weight",
    "explanation": "RMS Norm computes the root mean square of input along the feature dimension, then normalizes and scales."
  },

  "test_configs": [
    {"name": "batch_1", "batch_size": 1},
    {"name": "batch_2", "batch_size": 2},
    {"name": "batch_8", "batch_size": 8},
    {"name": "batch_512", "batch_size": 512}
  ],

  "baseline_ref": {
    "case_id": "rms_norm_hs4096_4096x512x1x1",
    "source": "ggml-python",
    "metric": "gbps",
    "note": "GGML baseline: ne=[hidden_size, 1, batch_size, 1]"
  }
}
```

### 多 hidden_size 变体

```json
{
  "name": "fp32_rms_norm_qwen2_5_7b_hs3584",
  "op_type": "rms_norm",
  "op_category": "normalization",
  "variant": "FP32",
  "description": "RMS Norm for Qwen2.5-7B with hidden_size=3584",
  "model_architectures": ["qwen2.5-7b"],

  "axes": {
    "batch_size": {"type": "var"},
    "hidden_size": {"type": "const", "value": 3584},
    "epsilon": {"type": "const", "value": 1e-6}
  },

  "inputs": {
    "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
    "weight": {"shape": ["hidden_size"], "dtype": "float32"}
  },

  "outputs": {
    "output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}
  },

  "baseline_ref": {
    "case_id": "rms_norm_hs3584_3584x512x1x1",
    "source": "ggml-python",
    "metric": "gbps"
  }
}
```

---

## 四、TopK Definition

### 完整示例

```json
{
  "name": "fp32_top_k_sampling_llama3_8b_k8",
  "op_type": "topk",
  "op_category": "sampling",
  "variant": "FP32",
  "description": "Top-k sampling from probabilities. Keeps only the k highest probability tokens, renormalizes, then samples from the filtered distribution. Used in LLaMA-3-8B decoding.",
  "tags": [
    "status:generated",
    "framework:ggml",
    "source:ggml-python",
    "layer:sampling",
    "sampling:top_k",
    "activation:fp32"
  ],
  "model_architectures": ["llama3-8b"],

  "axes": {
    "batch_size": {
      "type": "var",
      "description": "Batch dimension (number of parallel sequences)"
    },
    "vocab_size": {
      "type": "const",
      "value": 128256,
      "description": "Full vocabulary size"
    },
    "k": {
      "type": "const",
      "value": 8,
      "description": "Number of top tokens to select"
    }
  },

  "inputs": {
    "probs": {
      "shape": ["batch_size", "vocab_size"],
      "dtype": "float32",
      "description": "Probability distribution over vocabulary"
    },
    "top_k": {
      "shape": ["batch_size"],
      "dtype": "int32",
      "description": "K value per batch (typically constant)"
    }
  },

  "outputs": {
    "samples": {
      "shape": ["batch_size"],
      "dtype": "int64",
      "description": "Sampled token indices"
    },
    "top_indices": {
      "shape": ["batch_size", "k"],
      "dtype": "int32",
      "description": "Indices of top-k tokens (optional)"
    },
    "top_values": {
      "shape": ["batch_size", "k"],
      "dtype": "float32",
      "description": "Values of top-k tokens (optional)"
    }
  },

  "formula": {
    "computation": "top_k = argsort(probs, descending=True)[:k]; samples = categorical(top_k)",
    "explanation": "Select top-k highest probability tokens, renormalize, and sample from the filtered distribution."
  },

  "test_configs": [
    {"name": "batch_1", "batch_size": 1},
    {"name": "batch_8", "batch_size": 8},
    {"name": "batch_512", "batch_size": 512}
  ],

  "baseline_ref": {
    "case_id": "topk_k8_ne0256_256x512x1x1",
    "source": "ggml-python",
    "metric": "gbps",
    "note": "GGML baseline: ne=[vocab_subset, batch_size, 1, 1]"
  }
}
```

### 不同 k 值变体

```json
{
  "name": "fp32_top_k_sampling_llama3_8b_k6",
  "op_type": "topk",
  "op_category": "sampling",
  "variant": "FP32",
  "description": "Top-6 sampling for LLaMA-3-8B",
  "model_architectures": ["llama3-8b"],

  "axes": {
    "batch_size": {"type": "var"},
    "vocab_size": {"type": "const", "value": 128256},
    "k": {"type": "const", "value": 6}
  },

  "inputs": {
    "probs": {"shape": ["batch_size", "vocab_size"], "dtype": "float32"},
    "top_k": {"shape": ["batch_size"], "dtype": "int32"}
  },

  "outputs": {
    "samples": {"shape": ["batch_size"], "dtype": "int64"}
  },

  "baseline_ref": {
    "case_id": "topk_k6_ne0256_256x512x1x1",
    "source": "ggml-python",
    "metric": "gbps"
  }
}
```

---

## 五、Flash Attention Definition

### 完整示例

```json
{
  "name": "fp32_flash_attention_llama3_8b_cache512",
  "op_type": "flash_attention",
  "op_category": "attention",
  "variant": "FP32",
  "description": "Flash Attention for LLaMA-3-8B with KV cache size 512. Optimized attention computation that reduces memory accesses by tiling the attention matrix.",
  "tags": [
    "status:generated",
    "framework:ggml",
    "source:ggml-python",
    "layer:attention",
    "attention:flash",
    "kv_cache:f16"
  ],
  "model_architectures": ["llama3-8b"],

  "axes": {
    "batch_size": {
      "type": "var",
      "description": "Number of attention blocks (nb)"
    },
    "seq_len": {
      "type": "const",
      "value": 512,
      "description": "KV cache sequence length"
    },
    "num_heads": {
      "type": "const",
      "value": 32,
      "description": "Number of attention heads"
    },
    "head_dim": {
      "type": "const",
      "value": 128,
      "description": "Dimension per head"
    }
  },

  "inputs": {
    "query": {
      "shape": ["batch_size", "num_heads", "head_dim"],
      "dtype": "float32",
      "description": "Query projection"
    },
    "key_cache": {
      "shape": ["seq_len", "num_heads", "head_dim"],
      "dtype": "float16",
      "description": "Cached key projections"
    },
    "value_cache": {
      "shape": ["seq_len", "num_heads", "head_dim"],
      "dtype": "float16",
      "description": "Cached value projections"
    }
  },

  "outputs": {
    "output": {
      "shape": ["batch_size", "num_heads", "head_dim"],
      "dtype": "float32",
      "description": "Attention output"
    }
  },

  "formula": {
    "computation": "attn = softmax(Q @ K^T / sqrt(d)) @ V",
    "explanation": "Flash Attention computes attention by tiling the attention matrix to reduce HBM accesses."
  },

  "test_configs": [
    {"name": "batch_1", "batch_size": 1},
    {"name": "batch_8", "batch_size": 8},
    {"name": "batch_512", "batch_size": 512}
  ],

  "baseline_ref": {
    "case_id": "flash_attn_Llama3-8B_F16_cache512_nb512",
    "source": "ggml-python",
    "metric": "tflops",
    "note": "GGML baseline: model=Llama3-8B, kv_type=F16, cache_size=512, nb=batch_size"
  }
}
```

### 不同 cache size 变体

```json
{
  "name": "fp32_flash_attention_llama3_8b_cache4096",
  "op_type": "flash_attention",
  "op_category": "attention",
  "variant": "FP32",
  "description": "Flash Attention for LLaMA-3-8B with KV cache size 4096",
  "model_architectures": ["llama3-8b"],

  "axes": {
    "batch_size": {"type": "var"},
    "seq_len": {"type": "const", "value": 4096},
    "num_heads": {"type": "const", "value": 32},
    "head_dim": {"type": "const", "value": 128}
  },

  "inputs": {
    "query": {"shape": ["batch_size", "num_heads", "head_dim"], "dtype": "float32"},
    "key_cache": {"shape": ["seq_len", "num_heads", "head_dim"], "dtype": "float16"},
    "value_cache": {"shape": ["seq_len", "num_heads", "head_dim"], "dtype": "float16"}
  },

  "outputs": {
    "output": {"shape": ["batch_size", "num_heads", "head_dim"], "dtype": "float32"}
  },

  "baseline_ref": {
    "case_id": "flash_attn_Llama3-8B_F16_cache4096_nb512",
    "source": "ggml-python",
    "metric": "tflops"
  }
}
```

---

## 六、目录结构建议

```
definitions/
├── quant_gemm/                    # 现有 GEMM
│   ├── llama/
│   │   └── w4a32c8_q4_0_fp32_int8_llama3_8b_att_out_n4096_k4096.json
│   └── deepseek_v3/
│
├── rms_norm/                      # 新增
│   ├── llama/
│   │   ├── fp32_rms_norm_llama3_8b_hs4096.json
│   │   └── fp32_rms_norm_llama3_8b_hs2048.json
│   └── qwen/
│       └── fp32_rms_norm_qwen2_5_7b_hs3584.json
│
├── topk/                          # 新增
│   ├── llama/
│   │   ├── fp32_top_k_sampling_llama3_8b_k6.json
│   │   └── fp32_top_k_sampling_llama3_8b_k8.json
│   └── qwen/
│       └── fp32_top_k_sampling_qwen2_5_7b_k8.json
│
└── flash_attention/               # 新增
    ├── llama/
    │   ├── fp32_flash_attention_llama3_8b_cache512.json
    │   ├── fp32_flash_attention_llama3_8b_cache4096.json
    │   └── fp32_flash_attention_llama3_8b_cache8192.json
    └── qwen/
        ├── fp32_flash_attention_qwen2_5_7b_cache512.json
        └── fp32_flash_attention_qwen2_5_7b_cache4096.json
```

---

## 七、Baseline Case ID 映射表

| 算子 | Definition 参数 | Baseline Case ID 格式 |
|------|-----------------|----------------------|
| RMS Norm | hidden_size, batch_size | `rms_norm_hs{hidden_size}_{hidden_size}x{batch_size}x1x1` |
| TopK | k, vocab_size, batch_size | `topk_k{k}_ne0{vocab_size}_{vocab_size}x{batch_size}x1x1` |
| Flash Attn | model, kv_type, cache_size, nb | `flash_attn_{model}_{kv_type}_cache{cache_size}_nb{nb}` |

---

## 八、下一步实现

1. **创建 `NewOperatorsDefinitionGenerator`** - 从 baseline 自动生成 definition
2. **扩展 `DefinitionBaselineMapper`** - 支持新算子的 baseline 查询
3. **更新测试框架** - 支持不同性能指标 (TFLOPS vs GB/s)
