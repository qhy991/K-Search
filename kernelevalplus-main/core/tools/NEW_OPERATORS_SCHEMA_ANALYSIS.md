# 新算子 Schema 映射分析

## 一、现有 Schema 系统回顾

### Quantized GEMM 的定义结构
```json
{
  "op_type": "quant_gemm",
  "variant": "W4A32C8",
  "axes": {
    "M": {"type": "var", "description": "Batch dimension"},
    "N": {"type": "const", "value": 4096, "description": "Output features"},
    "K": {"type": "const", "value": 4096, "description": "Input features"}
  },
  "inputs": {
    "activation": {"shape": ["M", "K"], "dtype": "float32"},
    "weight": {"shape": ["N", "K/32"], "dtype": "block_q4_0"}
  },
  "outputs": {
    "output": {"shape": ["M", "N"], "dtype": "float32"}
  }
}
```

### Case ID 格式
- Definition: `w4a32c8_q4_0_fp32_int8_llama3_8b_att_out_n4096_k4096`
- Baseline: `w4a32c8_q4_0_f32_m4096_n1_k4096`

---

## 二、新算子分析

### 1. Flash Attention

#### 算子功能
Flash Attention 是 Transformer 模型中注意力机制的优化实现，用于计算 Query、Key、Value 的注意力分数和输出。

#### 维度参数
| 参数 | 含义 | 值域 |
|------|------|------|
| model | 模型名称 | Llama3-8B, Qwen2.5-7B, DeepSeekV2 |
| kv_type | KV缓存数据类型 | F16, Q8_0, Q4_0 |
| kv_cache_size | KV缓存序列长度 | 512, 4096, 8192 |
| nb (num_blocks) | 批处理的块数量 | 1, 2, 3, 4, 5, 8, 512 |

#### 映射到现有 Schema

```json
{
  "op_type": "flash_attention",
  "variant": "FP32",  // 或 "W4A32", "W8A32" 表示 KV 量化类型
  "model_architectures": ["llama3-8b"],
  "axes": {
    "seq_len": {"type": "const", "value": 512, "description": "KV cache sequence length"},
    "batch_size": {"type": "var", "description": "Number of blocks (nb)"}
  },
  "inputs": {
    "query": {"shape": ["batch_size", "seq_len", "head_dim"], "dtype": "float32"},
    "key_cache": {"shape": ["batch_size", "seq_len", "head_dim"], "dtype": "float16"},
    "value_cache": {"shape": ["batch_size", "seq_len", "head_dim"], "dtype": "float16"}
  },
  "outputs": {
    "output": {"shape": ["batch_size", "seq_len", "head_dim"], "dtype": "float32"}
  },
  "baseline_ref": {
    "case_id": "flash_attn_Llama3-8B_F16_cache512_nb512",
    "hardware": "A100",
    "metric": "tflops"
  }
}
```

#### Case ID 格式
- Baseline: `flash_attn_{model}_{kv_type}_cache{kv_cache_size}_nb{nb}`
- 示例: `flash_attn_Llama3-8B_F16_cache512_nb512`

---

### 2. RMS Norm (Root Mean Square Normalization)

#### 算子功能
RMS Norm 是一种归一化层，用于稳定神经网络训练。与 Layer Norm 类似，但计算更简单。

#### 维度参数
| 参数 | 含义 | 值域 |
|------|------|------|
| hidden_size | 隐藏层维度 | 128, 512, 1536, 2560, 3584, 4096, 5120, 7168 |
| ne | 输入张量形状 | [ne0, ne1, ne2, ne3] |

#### 实际含义解析
```
ne = [hidden_size, 1/8/32, 1-512, 1]
- ne0 = hidden_size: 特征维度
- ne1 = 1, 8, 32: 多头/分组数
- ne2 = 1-512: batch/序列长度
- ne3 = 1: 固定
```

#### 映射到现有 Schema

```json
{
  "op_type": "rms_norm",
  "variant": "FP32",
  "model_architectures": ["llama3-8b"],
  "axes": {
    "hidden_size": {"type": "const", "value": 4096, "description": "Feature dimension"},
    "batch_size": {"type": "var", "description": "Batch/sequence dimension"}
  },
  "inputs": {
    "input": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
    "weight": {"shape": ["hidden_size"], "dtype": "float32"}
  },
  "outputs": {
    "output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}
  },
  "formula": {
    "computation": "output = input / sqrt(mean(input^2) + eps) * weight"
  },
  "baseline_ref": {
    "case_id": "rms_norm_hs4096_4096x512x1x1",
    "hardware": "A100",
    "metric": "gbps"
  }
}
```

#### Case ID 格式
- Baseline: `rms_norm_hs{hidden_size}_{ne0}x{ne1}x{ne2}x{ne3}`
- 示例: `rms_norm_hs4096_4096x512x1x1`

---

### 3. TopK

#### 算子功能
TopK 选择输入张量中最大的 K 个值及其索引，常用于语言模型的采样（beam search, nucleus sampling）。

#### 维度参数
| 参数 | 含义 | 值域 |
|------|------|------|
| k | 选择的 Top-K 数量 | 6, 8 |
| ne0 | 第一维度（词汇表子集大小） | 160, 256 |
| ne | 输入张量形状 | [ne0, 1-512, 1, 1] |

#### 实际含义解析
```
ne = [ne0, ne1, ne2, ne3]
- ne0 = 160, 256: 词汇表子集大小
- ne1 = 1-512: batch size
- ne2 = 1: 固定
- ne3 = 1: 固定
```

#### 映射到现有 Schema

```json
{
  "op_type": "topk",
  "variant": "FP32",
  "model_architectures": ["llama3-8b"],
  "axes": {
    "vocab_size": {"type": "const", "value": 256, "description": "Vocabulary subset size"},
    "k": {"type": "const", "value": 8, "description": "Top-K value"},
    "batch_size": {"type": "var", "description": "Batch dimension"}
  },
  "inputs": {
    "input": {"shape": ["batch_size", "vocab_size"], "dtype": "float32"}
  },
  "outputs": {
    "values": {"shape": ["batch_size", "k"], "dtype": "float32"},
    "indices": {"shape": ["batch_size", "k"], "dtype": "int32"}
  },
  "baseline_ref": {
    "case_id": "topk_k8_ne0256_256x512x1x1",
    "hardware": "A100",
    "metric": "gbps"
  }
}
```

#### Case ID 格式
- Baseline: `topk_k{k}_ne0{ne0}_{ne0}x{ne1}x{ne2}x{ne3}`
- 示例: `topk_k8_ne0256_256x512x1x1`

---

## 三、扩展方案

### 方案A: 扩展 op_type 字段

推荐扩展 `op_type` 字段支持新算子：

```json
{
  "op_type": "flash_attention | rms_norm | topk | quant_gemm",
  ...
}
```

### 方案B: 统一 Performance Metric

在 baseline_ref 中明确指定性能指标：

```json
{
  "baseline_ref": {
    "metric": "tflops | gbps",  // tflops for compute-bound, gbps for memory-bound
    ...
  }
}
```

### 方案C: 扩展 DefinitionBaselineMapper

需要创建 `NewOperatorsBaselineMapper` 类来处理新算子：

```python
class NewOperatorsBaselineMapper:
    """新算子 Baseline 映射器"""

    # 算子类型映射
    OP_TYPE_MAPPING = {
        "flash_attention": "flash_attn",
        "rms_norm": "rms_norm",
        "topk": "topk"
    }

    def get_flash_attn_baseline(self, model, kv_type, cache_size, nb, hardware):
        case_id = f"flash_attn_{model}_{kv_type}_cache{cache_size}_nb{nb}"
        ...

    def get_rms_norm_baseline(self, hidden_size, ne, hardware):
        case_id = f"rms_norm_hs{hidden_size}_{'x'.join(map(str, ne))}"
        ...

    def get_topk_baseline(self, k, ne0, ne, hardware):
        case_id = f"topk_k{k}_ne0{ne0}_{'x'.join(map(str, ne))}"
        ...
```

---

## 四、Summary Table

| 算子 | op_type | 性能指标 | 关键维度 | Case ID 格式 |
|------|---------|----------|----------|--------------|
| Quantized GEMM | quant_gemm | TFLOPS | M×N×K | w4a32c8_q4_0_f32_m{N}_n{M}_k{K} |
| Flash Attention | flash_attention | TFLOPS | model×kv_type×cache×nb | flash_attn_{model}_{kv_type}_cache{size}_nb{nb} |
| RMS Norm | rms_norm | GB/s | hidden_size×ne | rms_norm_hs{size}_{ne0}x{ne1}x{ne2}x{ne3} |
| TopK | topk | GB/s | k×ne0×ne | topk_k{k}_ne0{ne0}_{ne0}x{ne1}x{ne2}x{ne3} |

---

## 五、建议的目录结构

```
definitions/
├── quant_gemm/           # 现有 GEMM 算子
│   ├── llama/
│   └── deepseek_v3/
├── flash_attention/      # 新增 Flash Attention
│   ├── llama/
│   │   └── fp32_flash_attn_llama3_8b_n512.json
│   └── qwen/
├── rms_norm/             # 新增 RMS Norm
│   ├── llama/
│   │   └── fp32_rms_norm_hs4096.json
│   └── deepseek_v3/
└── topk/                 # 新增 TopK
    ├── llama/
    │   └── fp32_topk_k8.json
    └── qwen/
```

---

## 六、下一步行动

1. 创建 `NewOperatorsBaselineMapper` 类
2. 为每个新算子生成示例 definition.json 文件
3. 更新 `generate_definitions_from_baseline.py` 支持新算子
4. 扩展测试框架支持新算子的性能测试
