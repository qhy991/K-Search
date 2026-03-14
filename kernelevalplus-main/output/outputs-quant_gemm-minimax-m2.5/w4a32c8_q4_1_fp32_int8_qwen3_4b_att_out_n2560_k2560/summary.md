# W4A32C8 Q4_1 量化 GEMM 优化总结

## 任务描述
- **算子**: 量化 GEMM (W4A32C8)
- **模型**: Qwen3-4B Attention Output 投影层
- **维度**: M=可变 (1-512), N=2560, K=2560
- **权重格式**: Q4_1 (4位量化，包含scale和min)
- **激活格式**: FP32 → Q8_1 动态量化
- **输出格式**: FP32

## 硬件环境
- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- SM Count: 128

## 优化历程

### 版本 v1 (初始版本)
- **方法**: 仅使用tiled kernel处理所有batch大小
- **结果**:
  - single_token: 0.276 ms, 0.047 TFLOPS
  - small_batch: 0.485 ms, 0.216 TFLOPS
  - large_batch: 0.666 ms, 10.072 TFLOPS
- **问题**: 单token性能较差

### 版本 v2
- **方法**: 尝试warp-level kernel优化小batch
- **结果**: 正确性测试失败 (NMSE > 0.05)
- **问题**: warp kernel存在计算错误

### 版本 v3 (最终版本) ✅
- **方法**:
  - M=1: 使用简单256线程kernel (无shared memory开销)
  - M≥2: 使用tiled kernel
- **结果**:
  - single_token: 0.046 ms, 0.287 TFLOPS (6.1x 提升)
  - small_batch: 0.486 ms, 0.216 TFLOPS
  - large_batch: 0.668 ms, 10.048 TFLOPS

## 性能对比

| 版本 | single_token | small_batch | large_batch |
|------|-------------|-------------|-------------|
| v1 | 0.276 ms, 0.047 TFLOPS | 0.485 ms, 0.216 TFLOPS | 0.666 ms, 10.072 TFLOPS |
| **v3** | **0.046 ms, 0.287 TFLOPS** | 0.486 ms, 0.216 TFLOPS | 0.668 ms, 10.048 TFLOPS |
| 提升 | **6.1x** | 相同 | 相同 |

## 优化技术

1. **DP4A指令**: 使用硬件INT8点积指令
2. **动态量化**: Q8_1逐块激活量化
3. **Kernel特化**:
   - M=1: 简单kernel，256线程/输出，无shared memory开销
   - M≥2: tiled kernel，使用shared memory复用数据
4. **循环展开**: #pragma unroll 提升指令级并行

## 正确性验证

| 测试用例 | NMSE | 阈值 | 状态 |
|----------|------|------|------|
| single_token (M=1) | 0.000034 | 0.05 | ✅ 通过 |
| small_batch (M=2-8) | 0.000227 | 0.05 | ✅ 通过 |
| large_batch (M=512) | 0.000141 | 0.05 | ✅ 通过 |

## 最佳版本
**版本 v3** - `quant-gemm-attempts/w4a32c8_q4_1_fp32_int8_qwen3_4b_att_out_n2560_k2560_v3/kernel.cu`

## Roofline分析
- 操作强度 (OI): ~640 FLOPs/Byte
- 拐点: ~82 FLOPs/Byte
- 结论: 计算受限 → 最大化算术吞吐量
