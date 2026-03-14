# Quant-GEMM Schema 命名规范

## 概述

本文档定义了量化GEMM操作的schema文件命名规范。

## 命名格式

```
{op}_{wXaYcZ}_{weight_quant}_{compute}_{model}_{layer}_{dims}.json
```

### 格式说明

| 组件 | 说明 | 示例 |
|------|------|------|
| `op` | 操作类型 | `quant_gemm` |
| `wX` | Weight量化位宽 | `w4`, `w8` |
| `aY` | Activation存储位宽 | `a16` (fp16), `a32` (fp32) |
| `cZ` | Compute计算位宽 | `c8` (int8), `c16` (fp16), `c32` (fp32) |
| `weight_quant` | Weight量化格式 | `q4_0`, `q8_0`, `q4_k` |
| `compute` | 计算精度 | `fp32`, `int8`, `int16` |
| `model` | 模型简称 | `ds2`, `ds3`, `qw7`, `llama` |
| `layer` | 层类型 | `att_out`, `att_qkv`, `ffn_up` |
| `dims` | 维度信息 | `n5120_k5120` |

## wXaYcZ 编码详解

### wX - Weight量化位宽

| 值 | 含义 | 典型格式 |
|----|------|---------|
| `w4` | 4-bit权重量化 | Q4_0, Q4_K |
| `w8` | 8-bit权重量化 | Q8_0 |
| `w16` | 16-bit权重 (未量化) | FP16/BF16 |

### aY - Activation存储位宽

| 值 | 含义 | 数据类型 |
|----|------|---------|
| `a16` | 16-bit激活存储 | FP16 |
| `a32` | 32-bit激活存储 | FP32 |

### cZ - Compute计算位宽

| 值 | 含义 | 数据类型 | 加速器支持 |
|----|------|---------|-----------|
| `c8` | 8-bit计算 | INT8 | Tensor Core, GPU |
| `c16` | 16-bit计算 | FP16 | Tensor Core, GPU |
| `c32` | 32-bit计算 | FP32 | GPU, CPU |

## 示例

### W4A32C32 (Weight 4-bit, Activation FP32, Compute FP32)

```
w4a32c32_q4_0_fp32_ds2_att_out_n5120_k5120.json
```

- **Weight**: Q4_0量化 (4-bit)
- **Activation存储**: FP32 (32-bit)
- **计算精度**: FP32 (32-bit)
- **模型**: DeepSeek-V2 (`ds2`)
- **层**: Attention Output (`att_out`)
- **维度**: [M, 5120] @ [5120, 5120]^T

### W8A32C8 (Weight 8-bit, Activation FP32, Compute INT8)

```
w8a32c8_q8_0_int8_qw7_ffn_gate_n18944_k3584.json
```

- **Weight**: Q8_0量化 (8-bit)
- **Activation存储**: FP32 (32-bit)
- **计算精度**: INT8 (8-bit)
- **模型**: Qwen2.5-7B (`qw7`)
- **层**: FFN Gate (`ffn_gate`)
- **维度**: [M, 3584] @ [18944, 3584]^T

## 变体类型矩阵

| Weight \ Activation/Compute | A16C16 | A32C16 | A32C32 | A32C8 |
|----------------------------|--------|--------|--------|-------|
| **W4 (Q4_X)** | W4A16C16 | W4A32C16 | W4A32C32 ✅ | W4A32C8 |
| **W8 (Q8_X)** | W8A16C16 | W8A32C16 | W8A32C32 | W8A32C8 ✅ |

## 层类型命名规范

### Attention层

| 层名 | 说明 | DeepSeek | Qwen |
|------|------|----------|------|
| `att_qkv` | QKV联合投影 | ✅ MLA | ❌ |
| `q_proj` | Query投影 | ❌ | ✅ GQA |
| `k_proj` | Key投影 | ❌ | ✅ GQA |
| `v_proj` | Value投影 | ❌ | ✅ GQA |
| `att_out` | Attention输出投影 | ✅ | ✅ |

### FFN/MoE层

| 层名 | 说明 | DeepSeek | Qwen |
|------|------|----------|------|
| `moe_up` | MoE上投影 | ✅ | ❌ |
| `moe_down` | MoE下投影 | ✅ | ❌ |
| `ffn_gate` | FFN Gate投影 | ❌ | ✅ SwiGLU |
| `ffn_up` | FFN Up投影 | ❌ | ✅ SwiGLU |
| `ffn_down` | FFN Down投影 | ❌ | ✅ SwiGLU |

## 模型简称映射

| 简称 | 完整模型 | hidden_size |
|------|----------|-------------|
| `ds2` | DeepSeek-V2 | 5120 |
| `ds3` | DeepSeek-V3 | 7168 |
| `qw7` | Qwen2.5-7B | 3584 |
| `llama` | Llama系列 | - |

## 维度命名规范

```
n{N}_k{K}
```

- `n{N}`: 输出特征维度 (矩阵乘法后的N维度)
- `k{K}`: 输入特征维度 (矩阵乘法前的K维度)

### 示例

| 文件名 | N | K | 说明 |
|--------|---|---|------|
| `n5120_k5120` | 5120 | 5120 | 方阵，如att_out |
| `n15360_k5120` | 15360 | 5120 | QKV联合投影 (3×5120) |
| `n512_k3584` | 512 | 3584 | K/V投影 (GQA) |
| `n18944_k3584` | 18944 | 3584 | FFN扩展 (SwiGLU) |

## 文件内容字段要求

每个schema JSON文件必须包含以下字段：

```json
{
  "name": "{wXaYcZ}_{weight_quant}_{compute}_{model}_{layer}_{dims}",
  "variant": "W{X}A{Y}C{Z}",
  "op_type": "quant_gemm",
  "description": "详细描述",
  "tags": [
    "layer:{layer_type}",
    "quantization:{quant_format}",
    "compute:{compute_precision}",
    ...
  ],
  "axes": {
    "N": {"value": N},
    "K": {"value": K},
    ...
  }
}
```

## 变体性能特性

| 变体 | 内存占用 | 计算速度 | 精度 | 适用场景 |
|------|----------|----------|------|----------|
| W4A32C32 | 低 | 基准 | 中等 | 内存受限 |
| W4A32C16 | 低 | 快 | 中等 | GPU推理 |
| W4A32C8 | 低 | 最快 | 中等 | 高性能推理 |
| W8A32C32 | 中 | 基准 | 高 | 高精度需求 |
| W8A32C16 | 中 | 快 | 高 | GPU推理 |
| W8A32C8 | 中 | 最快 | 高 | 高性能+高精度 |

## 迁移指南

### 旧命名 → 新命名

| 旧命名 | 新命名 | 变化说明 |
|--------|--------|----------|
| `w4a16_q4_0_fp32` | `w4a32c32_q4_0_fp32` | 明确C32表示FP32计算 |
| `w4a8_q4_0_q8_1` | `w4a16c16_q4_0_fp16` | 明确存储和计算都是FP16 |
| `w8a8_q8_0_q8_0` | `w8a16c16_q8_0_fp16` | 明确存储和计算都是FP16 |

### 批量重命名脚本

使用 `rename_schemas.py` 或 `fix_w8a32c8_files.py` 脚本进行批量重命名。

## 实施步骤

1. ✅ 创建新的schema模板
2. ✅ 使用新命名规范生成schemas
3. ✅ 修复现有文件字段
4. ⏳ 更新变体映射
5. ⏳ 更新测试脚本
6. ⏳ 更新用户文档
