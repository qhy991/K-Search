# W4A32C8 Q4_0 × FP32 Quantized GEMM - DeepSeek-V3

**任务ID**: `w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168`

## 快速概览

| 项目 | 值 |
|------|-----|
| 算子类型 | 量化 GEMM (Q4_0权重 × FP32激活) |
| 模型 | DeepSeek-V3 |
| 层 | Attention Output Projection |
| 维度 | M(1-512) × N=7168 × K=7168 |
| 最佳性能 | 1.977 TFLOPS (M=8) |

## 性能对比

| M | 本实现 | Best-Known | 差距 |
|---|--------|------------|------|
| 1 | 1.821 TFLOPS | 1.825 TFLOPS | 0.2% |
| 8 | 1.977 TFLOPS | 1.979 TFLOPS | 0.1% |
| 512 | 2.232 TFLOPS | 2.320 TFLOPS | 4% |

## 正确性

✅ 所有测试通过 (NMSE < 0.001)

## 文件说明

- `kernel.cu` - 最佳版本CUDA实现
- `test_results.json` - 详细测试结果
- `summary.md` - 完整优化历程文档（中文）

## 使用方法

```python
import torch
import w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_quant_gemm_test

output = w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_quant_gemm_test.forward(
    weight,     # [N, K/32*18] uint8
    activation, # [M, K] float32
    M, N, K
)
```

## 关键优化

1. **Split-K策略** (M ≤ 8) - 小批次性能提升142%
2. **DP4A指令** - 4倍点积吞吐量
3. **Q8_1动态量化** - on-the-fly激活量化
4. **自适应调度** - 根据批次大小选择最优策略

## 硬件要求

- CUDA 12.8+
- Compute Capability ≥ 6.1
- 推荐: RTX 4090 或同等GPU
