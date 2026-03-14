# test_operator.py 使用指南与问题分析

## 概述

`test_operator.py` 是一个通用的算子测试框架，用于测试已编译的 CUDA kernel 实现的正确性和性能。

## 功能特性

### 1. 自动化测试流程

`test_operator.py` 提供了完整的 CUDA kernel 测试流程：

```
生成测试数据 (FP32)
    ↓
量化输入数据
    ↓
运行参考实现 (Python)
    ↓
运行 CUDA kernel (GPU)
    ↓
对比结果 (计算 NMSE)
    ↓
输出测试报告
```

### 2. 基于 spec.json 的配置

测试框架从 `spec.json` 文件读取算子配置：

```json
{
  "name": "w8a8c8_q8_0_q8_1",
  "inputs": {
    "weight": {
      "dtype": "block_q8_0",
      "shape": ["N", "K/32", 34]
    },
    "activation": {
      "dtype": "float32",
      "shape": ["M", "K"]
    }
  },
  "reference": "reference.py:run",
  "test_configs": [
    {"name": "single_token", "M": 1, "N": 7168, "K": 7168},
    {"name": "att_qkv", "M": 1, "N": 21504, "K": 7168}
  ],
  "accuracy": {
    "metric": "nmse",
    "threshold": 0.05
  }
}
```

### 3. 支持的功能

- ✅ 自动生成测试数据
- ✅ 自动量化输入
- ✅ 调用 Python 参考实现
- ✅ 调用编译的 CUDA kernel
- ✅ 计算精度指标（NMSE）
- ✅ 性能基准测试（可选）
- ✅ 支持自定义测试配置

## 使用方法

### 基本用法

```bash
python test_operator.py <operator_name> <operator_folder> [options]
```

### 示例

```bash
# 测试 W8A8C8 Q8_0×Q8_1 算子
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1

# 使用自定义配置
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --config "M=1,N=512,K=512"

# 运行性能基准测试
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --benchmark
```

## 测试流程详解

### 1. 数据生成阶段

```python
# 根据 spec.json 生成测试数据
weight = torch.randn(N, K, dtype=torch.float32, device='cuda')
activation = torch.randn(M, K, dtype=torch.float32, device='cuda')
```

### 2. 量化阶段

```python
# 根据 dtype 自动选择量化函数
if dtype == "block_q8_0":
    weight_q = quantize_q8_0(weight)  # [N, K/32, 34]
```

### 3. 参考实现阶段

```python
# 调用 Python 参考实现
from reference import run
ref_output = run(activation, weight_q)  # [M, N]
```

### 4. CUDA Kernel 阶段

```python
# 调用编译的 CUDA kernel
output = gemm_w8a8c8_q8_0_q8_1(weight_q, activation, M, N, K)
```

### 5. 精度验证阶段

```python
# 计算 NMSE (Normalized Mean Squared Error)
mse = torch.mean((output - ref_output) ** 2)
signal_power = torch.mean(ref_output ** 2)
nmse = mse / signal_power

# 判断是否通过
passed = nmse <= threshold  # threshold = 0.05
```

## 当前问题分析

### 问题描述

**DeepSeek-V3 W8A8C8 Q8_0×Q8_1 算子的 CUDA kernel 测试失败**

```
[FAIL] single_token: CUDA error: misaligned address
[FAIL] att_qkv: CUDA error: misaligned address
[FAIL] moe_up: CUDA error: misaligned address
[FAIL] moe_down: CUDA error: misaligned address
```

### 问题定位

#### 1. Python 参考实现 ✅ 正常

```bash
# 使用 test_deepseek_v3_comprehensive.py 测试
python test_deepseek_v3_comprehensive.py

# 结果：所有测试通过
✓ llama.cpp 公式: 相对误差 0.45%
✓ att_out:  NMSE 5.67e-05
✓ att_qkv:  NMSE 5.78e-05
✓ moe_up:   NMSE 5.73e-05
✓ moe_down: NMSE 5.71e-05
```

**结论**：Python 参考实现是正确的，可以作为对比基准。

#### 2. CUDA Kernel ❌ 内存对齐错误

```bash
# 使用 test_operator.py 测试
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1

# 结果：CUDA kernel 失败
✗ CUDA error: misaligned address
```

**错误位置**：CUDA kernel 内部的内存访问

### 技术分析

#### 内存对齐问题的根源

1. **Q8_0 数据格式**
   ```
   每个 block: 34 字节
   - 2 字节: FP16 scale
   - 32 字节: 32 个 int8 量化值
   ```

2. **内存布局**
   ```
   weight_q: [N, K/32, 34] uint8
   stride: (34, 34, 1)
   ```

   问题：34 不是 2 的幂次，导致每个 block 的起始地址可能不对齐

3. **访问模式**
   ```cuda
   // 计算 block 偏移
   long long w_offset = ((long long)n * num_blocks + kb) * 34;

   // 读取 FP16 scale (可能未对齐)
   __half scale = *((__half*)&weight[w_offset]);  // ❌ 可能未对齐
   ```

#### 已尝试的修复方案

1. ✅ **逐字节读取 scale**
   ```cuda
   union {
       uint16_t u16;
       __half f16;
   } scale_union;
   scale_union.u16 = ((uint16_t)weight[w_offset + 0]) |
                     (((uint16_t)weight[w_offset + 1]) << 8);
   float d_w = __half2float(scale_union.f16);
   ```

2. ✅ **逐字节打包 int8 值**
   ```cuda
   int8_t tmp[4];
   for (int j = 0; j < 4; ++j) {
       tmp[j] = (int8_t)weight[w_offset + 2 + i * 4 + j];
   }
   w_qs_int[i] = (int)tmp[0] | ((int)tmp[1] << 8) |
                 ((int)tmp[2] << 16) | ((int)tmp[3] << 24);
   ```

3. ✅ **修复 vdr 参数**
   ```cuda
   constexpr int vdr = 8;  // 从 2 改为 8
   ```

4. ✅ **修复 clamp 函数**
   ```cuda
   a_int32 = (a_int32 < -128) ? -128 : ((a_int32 > 127) ? 127 : a_int32);
   ```

**结果**：所有修复都未能解决问题

### 可能的原因

1. **编译器优化问题**
   - 编译器可能对未对齐的访问进行了优化
   - 需要使用 `-O0` 或特定的对齐属性

2. **其他未发现的对齐问题**
   - activation 数组的访问
   - output 数组的写入
   - 局部变量的对齐

3. **CUDA 架构特定问题**
   - 不同 GPU 架构对对齐的要求不同
   - 可能需要针对特定架构优化

## 对比：两种测试方式

### test_deepseek_v3_comprehensive.py

| 特性 | 说明 |
|------|------|
| **测试对象** | Python 参考实现 |
| **运行环境** | 纯 Python/PyTorch |
| **编译需求** | ❌ 无需编译 |
| **测试内容** | Python 实现 vs FP32 基准 |
| **当前状态** | ✅ 已修复并通过 |
| **用途** | 验证算法正确性 |

### test_operator.py

| 特性 | 说明 |
|------|------|
| **测试对象** | CUDA kernel 实现 |
| **运行环境** | GPU (CUDA) |
| **编译需求** | ✅ 需要编译 C++/CUDA |
| **测试内容** | CUDA kernel vs Python 参考实现 |
| **当前状态** | ❌ CUDA kernel 有对齐错误 |
| **用途** | 验证 GPU 实现正确性和性能 |

## 调试建议

### 1. 使用 CUDA 调试工具

```bash
# 使用 cuda-memcheck 检测内存错误
cuda-memcheck python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1

# 使用 compute-sanitizer (新版本)
compute-sanitizer --tool memcheck python test_operator.py ...
```

### 2. 启用同步模式

```bash
# 立即捕获 CUDA 错误
export CUDA_LAUNCH_BLOCKING=1
python test_operator.py ...
```

### 3. 参考成功的实现

查看其他成功的 kernel 实现：
```bash
# W4A8 Q4_0×Q8_1 (已成功)
python test_operator.py w4a8_q4_0_q8_1 \
    operators/quant_gemm/variants/generic/w4a8_q4_0_q8_1
```

### 4. 简化测试

从最简单的情况开始：
```bash
# 最小维度测试
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --config "M=1,N=32,K=32"
```

## 当前建议

### 短期方案

**使用 Python 参考实现**进行算法验证和测试：

```bash
# 运行完整测试
python python/test_deepseek_v3_comprehensive.py

# 运行快速测试
python python/test_deepseek_v3_fast.py
```

优点：
- ✅ 已验证正确
- ✅ 结果稳定可靠
- ✅ 可用于算法开发和验证

缺点：
- ⚠️ 运行速度较慢（CPU）
- ⚠️ 无法测试 GPU 性能

### 长期方案

**修复 CUDA kernel 的对齐问题**：

1. 使用 cuda-memcheck 精确定位问题
2. 参考 llama.cpp 的实现
3. 考虑重新设计内存布局
4. 添加对齐属性和编译选项

## 总结

- ✅ **test_deepseek_v3_comprehensive.py** 已完全可用
- ❌ **test_operator.py** 对于 W8A8C8 算子当前不可用
- 🔧 需要进一步调试 CUDA kernel 的内存对齐问题
- 📊 Python 参考实现可以作为可靠的算法验证工具

## 相关文件

- `python/test_operator.py` - 通用测试框架
- `python/test_deepseek_v3_comprehensive.py` - Python 参考实现测试
- `python/test_deepseek_v3_fast.py` - 快速测试版本
- `core/operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1/`
  - `spec.json` - 算子规格定义
  - `reference.py` - Python 参考实现
  - `kernel.cu` - CUDA kernel 实现
  - `bindings.cpp` - Python 绑定
