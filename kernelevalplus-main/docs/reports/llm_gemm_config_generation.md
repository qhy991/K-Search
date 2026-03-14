# LLM 模型架构 GEMM 量化配置调研报告

## 概述

本报告总结了主流大语言模型（LLM）架构中使用的 GEMM（矩阵乘法）操作和量化格式，并基于 DeepSeek-V3 配置格式生成了相应的 JSON 配置文件。

## 调研的模型架构

### 1. LLaMA 系列

| 模型 | Hidden Size | FFN Intermediate | Attention Heads | Layers | 特点 |
|------|-------------|------------------|-----------------|--------|------|
| LLaMA-2-7B | 4096 | 11008 | 32 | 32 | 标准Transformer |
| LLaMA-3-8B | 4096 | 14336 | 32 | 32 | 更大FFN (3.5x) |
| LLaMA-3-70B | 8192 | 28672 | 64 | 80 | 大模型 |

### 2. Mistral 系列

| 模型 | Hidden Size | FFN Intermediate | Attention Heads | Layers | 特点 |
|------|-------------|------------------|-----------------|--------|------|
| Mistral-7B | 4096 | 14336 | 32 | 32 | GQA (8 KV heads) |
| Mixtral-8x7B | 4096 | 14336 | 32 | 32 | MoE (8 experts, top-2) |

### 3. Qwen 系列

| 模型 | Hidden Size | FFN Intermediate | Attention Heads | Layers | 特点 |
|------|-------------|------------------|-----------------|--------|------|
| Qwen2.5-7B | 3584 | 18944 | 28 | 28 | YARN 位置编码 |
| Qwen2.5-32B | 5120 | 27392 | 40 | 64 | 更大容量 |
| Qwen3-0.6B | 1024 | 3072 | 16 | 28 | 边缘优化 |
| Qwen3-1.5B | 2560 | 9728 | 32 | 36 | 紧凑设计 |
| Qwen3-4B | 4096 | 12288 | 32 | 36 | 标准尺寸 |
| Qwen3-8B | 4096 | 12288 | 32 | 36 | 主流模型 |
| Qwen3-14B | 5120 | 17408 | 40 | 40 | 更大容量 |
| Qwen3-32B | 5120 | 25600 | 64 | 64 | 深层架构 (5x FFN) |

### 4. Gemma 系列

| 模型 | Hidden Size | FFN Intermediate | Attention Heads | Layers | 特点 |
|------|-------------|------------------|-----------------|--------|------|
| Gemma-2-9B | 3584 | 14336 | 16 | 42 | MQA, 滑动窗口 |
| Gemma-2-27B | 4608 | 36864 | 32 | 42 | 更大FFN (8x) |

### 5. Phi 系列

| 模型 | Hidden Size | FFN Intermediate | Attention Heads | Layers | 特点 |
|------|-------------|------------------|-----------------|--------|------|
| Phi-3-mini | 3072 | 8192 | 32 | 32 | 紧凑设计 |
| Phi-3-medium | 4096 | 16384 | 32 | 40 | 中等规模 |

## 量化格式说明

基于 llama.cpp 的量化格式：

### W4A8 (Q4_0 × Q8_1)
- **权重**: Q4_0 格式，18 字节/块 (32 值)
  - scale: FP16 (2 字节)
  - qs: uint8[16] (16 字节，32×4-bit 打包)
  - 公式: `w = (qs - 8) * scale`

- **激活**: Q8_1 格式，36 字节/块
  - d (scale): FP16 (2 字节)
  - s (sum): FP16 (2 字节)
  - qs: int8[32] (32 字节)

### W4A16 (Q4_0 × FP16)
- **权重**: Q4_0 格式（同上）
- **激活**: FP16 直接计算
- **计算**: 反量化后使用 FP16 GEMM

### W5.1A8 (Q5_1 × Q8_1)
- **权重**: Q5_1 格式，24 字节/块
  - d (scale): FP16 (2 字节)
  - m (min): FP16 (2 字节)
  - qh (high bits): uint8[4] (4 字节)
  - qs (low bits): uint8[16] (16 字节)
  - 非对称量化: `q = round((x - m) / d), x = q * d + m`

### W8A8 (Q8_0 × Q8_0)
- **权重**: Q8_0 格式，34 字节/块
  - scale: FP16 (2 字节)
  - qs: int8[32] (32 字节)
- **激活**: 动态量化到 Q8_0
- **计算**: INT8 点积

## 已生成的配置文件

### LLaMA 系列
```
definitions/quant_gemm/llama/
├── w4a8_q4_0_q8_1_llama3_8b_att_qkv_n12288_k4096.json
└── w8a8_q8_0_q8_0_llama3_70b_att_out_n8192_k8192.json
```

### Mistral 系列
```
definitions/quant_gemm/mistral/
└── w4a8_q4_0_q8_1_mistral7b_moe_up_n14336_k4096.json
```

### Mixtral 系列
```
definitions/quant_gemm/mixtral/
└── w8a8c8_q8_0_q8_1_mixtral8x7b_moe_up_n14336_k4096.json
```

### Qwen 系列
```
definitions/quant_gemm/qwen2_5_7b/
└── w4a8_q4_0_q8_1_qwen2_5_7b_att_qkv_n10752_k3584.json

definitions/quant_gemm/qwen3/
├── w4a16_f16_fp16_qwen3_0_6b_att_qkv_n3072_k1024.json
├── w4a8_q4_0_q8_1_qwen3_1_5b_att_qkv_n7680_k2560.json
├── w4a8_q4_0_q8_1_qwen3_8b_att_qkv_n12288_k4096.json
├── w8a8c8_q8_0_q8_1_qwen3_14b_moe_up_n17408_k5120.json
└── w5_1a8_q5_1_q8_1_qwen3_32b_ffn_up_n25600_k5120.json
```

### Gemma 系列
```
definitions/quant_gemm/gemma/
└── w5_1a8_q5_1_q8_1_gemma2_9b_ffn_up_n14336_k3584.json
```

### Phi 系列
```
definitions/quant_gemm/phi/
└── w4a16_f16_fp16_phi3_mini_ffn_up_n8192_k3072.json
```

## 配置文件格式规范

每个配置文件包含以下关键字段：

### 基本信息
- `name`: 配置文件名
- `op_type`: 操作类型 ("quant_gemm")
- `variant`: 量化变体 (W4A8, W8A8, W5.1A8 等)
- `description`: 详细描述

### 标签
- `status`: 验证状态
- `framework`: 框架 (llama, mistral, qwen 等)
- `source`: 来源 (llama.cpp)
- `layer`: 层类型 (attention-qkv, ffn-up, moe-up 等)
- `quantization`: 量化格式

### 轴定义
- `M`: 批次维度 (变长)
- `N`: 输出特征数 (常数)
- `K`: 输入特征数 (常数)
- `block_size`: 量化块大小 (32)

### 输入输出
- 输入激活和权重的形状、数据类型
- 输出张量的形状和数据类型

### 类型定义
- 量化格式的详细字段定义
- 字节大小和量化公式

### 约束条件
- 维度对齐要求
- 尺寸约束

### 计算公式
- 计算流程说明
- llama.cpp 兼容公式

### 测试配置
- single_token: M=1
- small_batch: M=16
- medium_batch: M=128
- large_batch: M=512

### 性能说明
- 量化格式的好处
- 内存/计算边界分析

## GEMM 操作类型

### Attention 相关
1. **QKV Projection**: `[M, hidden] @ [3*hidden, hidden]^T → [M, 3*hidden]`
   - LLaMA-3-8B: N=12288, K=4096
   - Qwen2.5-7B: N=10752, K=3584

2. **Attention Output**: `[M, hidden] @ [hidden, hidden]^T → [M, hidden]`
   - LLaMA-3-70B: N=8192, K=8192

### FFN 相关
1. **FFN Up**: `[M, hidden] @ [ffn_dim, hidden]^T → [M, ffn_dim]`
   - LLaMA-3-8B: N=14336, K=4096 (3.5x)
   - Gemma-2-9B: N=14336, K=3584 (4x)
   - Phi-3-mini: N=8192, K=3072 (2.67x)

### MoE 相关
1. **MoE Up**: 同 FFN Up，但 M 包含 expert 维度
   - Mixtral-8x7B: N=14336, K=4096 (8 experts, top-2)

2. **MoE Down**: `[M*experts, ffn_dim] @ [hidden, ffn_dim]^T → [M*experts, hidden]`
   - DeepSeek-V3: N=7168, K=18432

## 量化选择建议

### 精度 vs 性能权衡

| 量化格式 | 内存压缩 | 计算速度 | 精度损失 | 推荐场景 |
|---------|---------|---------|---------|---------|
| W4A8 | 4x | 快 (INT32) | 中等 | 边缘设备，内存受限 |
| W4A16 | 4x | 中等 (FP16) | 较小 | 需要较好精度的场景 |
| W5.1A8 | 3.2x | 快 (INT32) | 较小 | 平衡精度和性能 |
| W8A8 | 2x | 最快 (INT8) | 最小 | 高性能场景 |

### 模型特定建议

1. **小型模型 (≤7B)**: W4A8 适合边缘部署
2. **中型模型 (7B-13B)**: W5.1A8 或 W8A8 平衡性能
3. **大型模型 (≥30B)**: W8A8 保证精度
4. **MoE 模型**: W8A8 用于 expert 计算
5. **Edge 设备**: W4A16 配合 FP16 计算

## 参考资源

### 模型架构
- [LLaMA Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [Mixtral 8x7B Paper](https://arxiv.org/pdf/2401.04088)
- [Gemma 2 Technical Report](https://arxiv.org/html/2408.00118v1)
- [Qwen2.5 Technical Report](https://arxiv.org/pdf/2409.12186)

### 量化格式
- [llama.cpp Quantization Discussion #5063](https://github.com/ggml-org/llama.cpp/discussions/5063)
- [Which Quantization Should I Use?](https://arxiv.org/html/2601.14277v1)
- [GGUF Optimization Deep Dive](https://medium.com/@michael.hannecke/gguf-optimization-a-technical-deep-dive-for-practitioners-ce84c8987944)

## 下一步工作

1. 为每个模型的每个层类型生成完整配置
2. 添加更多量化变体 (W4.1A8, W6A8 等)
3. 验证配置与实际模型的匹配
4. 生成对应的内核实现代码
5. 添加性能基准测试配置

## 配置文件命名规范

```
{variant}_{model}_{layer}_n{N}_k{K}.json
```

示例:
- `w4a8_q4_0_q8_1_llama3_8b_att_qkv_n12288_k4096.json`
- `w8a8c8_q8_0_q8_1_mixtral8x7b_moe_up_n14336_k4096.json`

其中:
- `{variant}`: 量化变体 (w4a8, w8a8, w5_1a8 等)
- `{model}`: 模型名称 (llama3_8b, mistral7b 等)
- `{layer}`: 层类型 (att_qkv, ffn_up, moe_down 等)
- `n{N}`: 输出维度
- `k{K}`: 输入维度
