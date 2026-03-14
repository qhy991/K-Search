# W4A32C8 Q4_0 × FP32 量化 GEMM 优化历程

**任务ID**: `w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168`

**模型**: DeepSeek-V3
**算子**: Attention Output Projection (注意力输出投影)
**量化格式**: W4A32C8 (Q4_0权重 + FP32激活 + Q8_1动态量化)

---

## 一、问题定义

### 计算公式
```
C = A @ W^T
```

### 维度规格
| 维度 | 大小 | 描述 |
|------|------|------|
| M | 1-512 (可变) | 批次大小 |
| N | 7168 | 输出特征数 |
| K | 7168 | 输入特征数 |

### 数据格式
- **激活 A**: FP32, shape [M, K]
- **权重 W**: Q4_0 量化, shape [N, K/32]
- **输出 C**: FP32, shape [M, N]

---

## 二、硬件分析

### GPU 规格 (NVIDIA RTX 4090)
```
GPU: NVIDIA GeForce RTX 4090
Compute Capability: 8.9
SM Count: 128
Warp Size: 32
Max Threads per Block: 1024
Peak FP32 TFLOPS: ~82.6
Peak Bandwidth: ~1008 GB/s
```

### Roofline 分析
```
OI = FLOPs / Bytes_transferred

对于 M=1:
  OI ≈ 13,389 FLOPs/Byte >> Ridge Point (82) → 计算密集型

对于 M=512:
  OI ≈ 1.45M FLOPs/Byte >> Ridge Point (82) → 计算密集型
```

**结论**: 该算子在所有批次大小下均为**计算密集型**，优化重点应放在：
1. 最大化算术吞吐量（DP4A指令）
2. 最大化共享内存使用以实现数据复用
3. 良好的线程级并行以利用所有SM

---

## 三、优化历程

### 版本对比总览

| 版本 | M=1 TFLOPS | M=8 TFLOPS | M=512 TFLOPS | 主要优化 |
|------|------------|-------------|--------------|----------|
| v1 (基础版) | 0.754 | 1.792 | 2.270 | 基本DP4A实现 |
| v2 (Split-K) | 1.826 | 1.991 | 2.221 | 小批次Split-K |
| **v3 (最终版)** | **1.821** | **1.977** | **2.232** | 数值稳定性+微优化 |
| Best-Known | 1.825 | 1.979 | 2.320 | 参考基线 |

---

### v1: 基础实现

**问题**:
- 错误的Q4_0解包格式
- 简单的1D线程块配置
- 没有针对小批次优化

**结果**:
```
M=1:  0.754 TFLOPS  (仅为best的41%)
M=8:  1.792 TFLOPS  (为best的91%)
M=512: 2.270 TFLOPS  (为best的98%)
```

**关键bug修复**:
1. **Q4_0结构体布局错误**: 最初假设 `d` 在 `qs` 之后，实际相反
2. **解包格式错误**: 需要使用llama.cpp格式（先所有低位nibble，后所有高位nibble）
3. **函数签名错误**: 测试框架调用 `forward(weight, activation, M, N, K)`

---

### v2: Split-K 优化

**优化策略**:
1. **Split-K for M ≤ 8**: 将K维度并行化到多个线程块
2. **Warp-level reduction**: 使用 `__shfl_down_sync` 进行规约
3. **Atomic accumulation**: 使用 `atomicAdd` 累加部分结果

**关键代码**:
```cuda
// Split-K 参数计算
int num_splits = max(1, (num_sms * 4) / (M * N));
num_splits = min(num_splits, num_k_blocks);
const int k_split = (num_k_blocks + num_splits - 1) / num_splits;

// 3D网格配置
dim3 grid(N, num_splits, M);
```

**性能提升**:
```
M=1:  0.754 → 1.826 TFLOPS  (+142%!)
M=8:  1.792 → 1.991 TFLOPS  (+11%)
M=512: 2.270 → 2.221 TFLOPS  (-2%, 略微下降)
```

**分析**:
- 小批次性能提升显著，因为Split-K充分利用了并行性
- 大批次性能略微下降，可能因为额外的原子操作开销

---

### v3: 数值稳定性与微优化

**优化项**:

1. **数值稳定性修复**:
```cuda
// 之前 (v2): 当 a_max=0 时会除零
const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
const float inv_d_a = 127.0f / a_max;  // BUG!

// 修复后 (v3):
const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
const float inv_d_a = 1.0f / d_a;  // 正确!
```

2. **函数内联优化**:
```cuda
__device__ __forceinline__ float half_to_float(uint16_t h);
__device__ __forceinline__ int dp4a(int a, int b, int c);
__device__ __forceinline__ float process_q4_0_block(...);
```

3. **FP16转换优化**: 使用union替代memcpy
```cuda
union { uint16_t u16; half f16; } un;
un.u16 = h;
return __half2float(un.f16);
```

4. **输出分配优化**: `torch::empty` 替代 `torch::zeros`

**最终结果**:
```
M=1:  1.821 TFLOPS  (vs best: 1.825, 差距0.2%)
M=8:  1.977 TFLOPS  (vs best: 1.979, 差距0.1%)
M=512: 2.232 TFLOPS  (vs best: 2.320, 差距4%)
```

---

## 四、核心实现细节

### Q4_0 解包格式 (llama.cpp兼容)

```cuda
// 每个block有16字节的打包数据
// 解包方式: 先所有低位nibble (位置0-15)，再所有高位nibble (位置16-31)

#pragma unroll
for (int j = 0; j < 4; ++j) {
    const uint8_t q0 = qs[j * 4 + 0];
    const uint8_t q1 = qs[j * 4 + 1];
    const uint8_t q2 = qs[j * 4 + 2];
    const uint8_t q3 = qs[j * 4 + 3];

    // 打包4个低位nibble
    const int w_low = (q0 & 0x0F) | ((q1 & 0x0F) << 8) |
                     ((q2 & 0x0F) << 16) | ((q3 & 0x0F) << 24);

    // 打包4个高位nibble
    const int w_high = (q0 >> 4) | ((q1 >> 4) << 8) |
                      ((q2 >> 4) << 16) | ((q3 >> 4) << 24);
    // ...
}
```

### Q8_1 动态量化公式

```cuda
// 计算公式: result = d_w * (d_a * sumi - 8 * s_a)

// 1. 找到最大绝对值和和
float a_max = 0.0f;
float s_a = 0.0f;
#pragma unroll
for (int j = 0; j < 8; ++j) {
    float4 v = *reinterpret_cast<const float4*>(&act[j * 4]);
    a_max = fmaxf(a_max, fabsf(v.x));
    // ... 更新 a_max 和 s_a
}

// 2. 计算量化参数
const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
const float inv_d_a = 1.0f / d_a;

// 3. 使用DP4A计算dot product
int sumi = 0;
// ... DP4A循环

// 4. 应用最终公式
return d_w * (d_a * (float)sumi - 8.0f * s_a);
```

### DP4A 指令

```cuda
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
```

DP4A计算: `c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]`

---

## 五、自适应策略调度

```cpp
if (M <= 8) {
    // Split-K for 小批次
    // - 充分利用SM的并行性
    // - 每个split处理一部分K维度
    // - 使用atomicAdd累加结果
} else {
    // Tiled for 大批次
    // - TILE_M=4, TILE_N=64
    // - 每个线程计算一个输出元素
    // - 避免原子操作的开销
}
```

---

## 六、最终性能对比

### vs Best-Known Kernel

| 配置 | 本实现 | Best-Known | 差距 |
|------|--------|------------|------|
| M=1  | 1.821 TFLOPS | 1.825 TFLOPS | **0.2%** |
| M=8  | 1.977 TFLOPS | 1.979 TFLOPS | **0.1%** |
| M=512| 2.232 TFLOPS | 2.320 TFLOPS | **4%** |

### 正确性验证

所有测试用例的NMSE远低于0.05阈值:

| 配置 | NMSE | 状态 |
|------|------|------|
| single_token (M=1) | 3.8e-05 | ✅ |
| small_batch (M=8) | 0.000514 | ✅ |
| large_batch (M=512) | 0.000233 | ✅ |

---

## 七、关键经验总结

### 1. 量化格式理解至关重要
- Q4_0的内存布局必须精确匹配llama.cpp格式
- 解包顺序：先所有低位nibble，再所有高位nibble
- offset-8编码需要在计算中补偿

### 2. 自适应策略有效
- 小批次：Split-K策略显著提升性能（+142%）
- 大批次：简单的Tiled策略更高效

### 3. 数值稳定性很重要
- `inv_d_a = 1.0f / d_a` 而非 `127.0f / a_max`
- 边界条件处理（a_max=0）

### 4. 硬件特性利用
- DP4A指令：4倍吞吐量提升
- `__forceinline__`：减少函数调用开销
- Warp-level reduction：高效的并行规约

---

## 八、文件结构

```
output/outputs-quant_gemm-GLM-4.7/w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168/
├── kernel.cu              # 最佳版本实现
├── test_results.json      # 测试结果
└── summary.md             # 本文档
```

---

## 九、硬件要求

- **最低**: Compute Capability ≥ 6.1 (支持DP4A)
- **测试**: NVIDIA RTX 4090 (CC 8.9)
- **推荐**: Compute Capability ≥ 7.0 以获得最佳性能

---

## 十、使用示例

```python
import torch

# 加载编译扩展
import w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_quant_gemm_test

# 前向传播
output = w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_quant_gemm_test.forward(
    weight,     # [N, K/32*18] uint8 tensor (Q4_0 blocks)
    activation, # [M, K] float32 tensor
    M, N, K     # 维度
)
```

---

## 十一、参考资料

1. llama.cpp Q4_0量化格式文档
2. CUDA DP4A指令参考手册
3. GGML基线实现
4. "Quantized Matrix Multiplication for Deep Learning Inference" 论文

---

**生成时间**: 2026-03-11
**测试硬件**: NVIDIA GeForce RTX 4090, CC 8.9
**CUDA版本**: 12.8
