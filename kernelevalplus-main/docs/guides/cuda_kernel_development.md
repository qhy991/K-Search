---
name: cuda-kernel-development
description: CUDA kernel 开发、测试与优化完整流程指导
version: 2.0.0
author: Agent4Kernel
tags: [code-generation, optimization, correctness, iterative-fix, cuda, quantization, testing]
---

# CUDA Kernel 开发与优化 Skill

本 Skill 提供完整的 CUDA kernel 开发流程指导，专注于量化 GEMM 操作。

## ⚙️ 环境配置

**重要**: 所有命令必须在正确的 Conda 环境中执行！

### 激活环境

```bash
# 必须先激活 Conda 环境
conda activate KM-12.8
```

### 环境要求

- **Conda 环境**: `KM-12.8` (包含 PyTorch 和 CUDA 支持)
- **Python**: 3.8+
- **PyTorch**: 2.0+ (JIT 编译需要)
- **CUDA**: 12.1+

### 设置项目路径

```bash
# 设置 KernelEvalPlus 项目根目录（根据实际环境调整）
export KERNEL_EVAL_PLUS_DIR=/path/to/kernelevalplus  # 修改为实际路径

# 进入项目目录
cd $KERNEL_EVAL_PLUS_DIR
```

### 验证环境

```bash
# 激活环境后验证
conda activate KM-12.8

# 检查 Python
python --version

# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查 CUDA
nvcc --version
```

## 核心能力

- **代码生成指导**：基于量化格式定义生成 CUDA kernel
- **自动化测试**：使用 test_runner.py 进行编译、正确性、性能测试
- **迭代修复机制**：根据测试结果自动修复问题
- **硬件优化**：结合 GPU 架构特性进行性能优化
- **Baseline 对比**：与 GGML baseline 数据库进行性能对比

## 开发流程

### 阶段 1: 理解任务需求

从定义文件中提取关键信息：

```json
{
  "name": "w4a32c8_q4_0_fp32_int8_ds3_lm_head_n102400_k5120",
  "axes": {
    "M": {"type": "var", "description": "Batch dimension"},
    "N": {"type": "const", "value": 102400, "description": "Output features"},
    "K": {"type": "const", "value": 5120, "description": "Input features"}
  },
  "inputs": {
    "weight": {"dtype": "q4_0", "shape": ["N", "K/32"]},
    "activation": {"dtype": "float32", "shape": ["M", "K"]}
  },
  "outputs": {
    "output": {"dtype": "float32", "shape": ["M", "N"]}
  },
  "formula": "result = d4_0 * (d_a * sumi - 8 * s_a)",
  "test_configs": [
    {"name": "batch_1", "M": 1, "N": 102400, "K": 5120},
    {"name": "batch_512", "M": 512, "N": 102400, "K": 5120}
  ]
}
```

**关键参数提取**:
- 量化格式：weight=q4_0, activation=fp32
- 维度：N=102400, K=5120 (宽高比 20:1)
- 计算公式：Q4_0 × FP32_INT8
- 测试场景：M=1 (单 token), M=512 (大 batch)

### 阶段 2: 检测硬件特性

使用 `core/tools/gpu_specs.py` 自动检测 GPU：

```bash
python -c "
from core.tools.gpu_specs import detect_gpu_hardware, GPU_SPEC_INFO
gpu = detect_gpu_hardware()
print(f'GPU: {gpu}')
if gpu in GPU_SPEC_INFO:
    spec = GPU_SPEC_INFO[gpu]
    print(f'Architecture: {spec[\"architecture\"]}')
    print(f'SM Count: {spec[\"sm_count\"]}')
    print(f'Memory Bandwidth: {spec[\"memory_bandwidth\"]} GB/s')
"
```

### 阶段 3: 生成 CUDA Kernel

#### 3.1 代码结构要求

生成的 `kernel.cu` 必须包含：

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>  // REQUIRED

// 量化类型定义
typedef struct {
    uint16_t d;        // scale stored as uint16_t (raw FP16 bits)
    int8_t qs[32];     // quantized values
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "");

// FP16 转换辅助函数 (CRITICAL)
__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// 主 kernel 函数
__global__ void gemm_kernel(
    const void* __restrict__ weight,
    const void* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    // 实现 kernel 逻辑
    // ...
}

// PyTorch 绑定 (REQUIRED)
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // kernel launch
    dim3 block(32, 32);  // blockDim.x * blockDim.y MUST be ≤ 1024
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM forward pass");
}
```

#### 3.2 关键约束

1. **blockDim.x * blockDim.y ≤ 1024** (CUDA 硬件限制)
2. **必须使用 `uint16_t` 存储 scale**，不能使用 `half` 类型
3. **必须使用 `read_half_as_float()` 辅助函数**转换 FP16
4. **必须包含 `#include <torch/extension.h>` 和 `PYBIND11_MODULE`**
5. **量化块大小必须是 32**

#### 3.3 量化格式公式

| 格式 | 公式 | 说明 |
|------|------|------|
| Q4_0 × Q8_1 | `d4_0 * (d8_1 * sumi - 8 * s8_1)` | Q4_0 使用 offset-8 编码 |
| Q8_0 × Q8_0 | `d8_0 * d_a * sumi` | 对称量化 |
| Q8_0 × FP32 | `d8_0 * d_a * sumi` | 激活动态量化到 Q8_0 |

#### 3.4 硬件优化策略

根据检测到的 GPU 架构选择优化策略：

**Hopper (H100)** - SM 10.0:
- 使用 TMA (Tensor Memory Accelerator) 进行异步数据传输
- Thread Block Clusters 提高并发度
- 大 L2 Cache 优化数据重用

**Ada (RTX 4090, L40S)** - SM 8.9:
- cp.async + Large Shared Memory (100KB)
- L2 Cache Persistence 控制
- FP8/TF32 Tensor Cores

**Ampere (A100, A800)** - SM 8.0:
- cp.async + Double Buffering
- L2 Residency Control
- TF32 Tensor Cores

**Turing (T4)** - SM 7.5:
- 向量化加载 (float4)
- Shared Memory 优化
- INT8/FP16 Tensor Cores

### 阶段 4: 提交代码进行测试

#### 4.1 创建测试目录

**目录结构：只需要 kernel.cu + definition JSON**

```bash
# 最简目录（推荐）：只需要 kernel.cu，通过 --definition 指定定义文件
<any_directory>/
├── kernel.cu           # 生成的代码（必须）
└── test_results.json   # 测试结果（自动生成）

# 传统目录（如果不使用 --definition）
llm_kernel_test/sandbox/generated/<definition_name>/<attempt_timestamp>/<variant>/
├── kernel.cu           # 生成的代码
├── spec.json           # 必须（如果不使用 --definition）
├── reference.py        # 可选（有内置参考实现）
└── test_results.json   # 测试结果（自动生成）
```

#### 4.2 使用 test_runner.py 测试

**推荐方式：使用 --definition 参数（无需 spec.json）**

```bash
# 激活环境
conda activate KM-12.8
cd $KERNEL_EVAL_PLUS_DIR

# 直接测试 kernel（只需 kernel.cu + definition JSON）
python llm_kernel_test/test_runner.py \
    --test \
    --variant <variant_name> \
    --attempt-path <path_to_kernel_dir> \
    --definition <path_to_definition.json>
```

`--definition` 参数会自动完成以下转换：
- **dtype 映射**：`q4_0` → `block_q4_0`，`q8_0` → `block_q8_0` 等
- **维度提取**：从 `axes` 中提取常量维度（N, K）作为默认值
- **参考实现**：自动选择内置的参考实现（Q4_0+FP32, Q4_0+Q8_1, Q8_0+FP32 等）
- **测试配置**：直接使用 definition 中的 `test_configs`

**示例**：
```bash
# 🌟 推荐：使用 --definition 参数（最简洁）
python llm_kernel_test/test_runner.py \
    --test \
    --variant W4A32C8 \
    --attempt-path /path/to/my_kernel_dir \
    --definition definitions/quant_gemm/deepseek_v3/w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168.json

# 传统方式：使用 --submit + --test（需要 spec.json 在 attempt 目录中）
python llm_kernel_test/test_runner.py \
    --submit my_kernel.cu \
    --variant w4a32c8_q4_0_fp32_int8 \
    --attempt-id deepseek_v2_lm_head_v1

python llm_kernel_test/test_runner.py \
    --test \
    --variant w4a32c8_q4_0_fp32_int8 \
    --attempt-id deepseek_v2_lm_head_v1
```

### 阶段 5: 迭代修复

test_runner.py 会自动进行三阶段测试：

#### 5.1 编译检查
- 使用 PyTorch JIT 编译
- 自动检测并修复常见问题：
  - 缺少 PYBIND11_MODULE → 自动生成 wrapper
  - block size 超过 1024 → 自动调整
  - FP16 转换错误 → 提供诊断信息

**编译错误诊断**：
```python
# 查看诊断信息
results["compilation"]["diagnostics"]
```

#### 5.2 正确性测试
- 对比 reference.py 计算参考输出
- 计算 NMSE（归一化均方误差）
- 正确性标准：**NMSE ≤ 0.1**

#### 5.3 性能测试与 Baseline 对比
- Benchmark 延迟和 FLOPS
- 自动与 GGML baseline 对比
- 显示性能比率

**性能评估标准**：
- ✅ 超越基线: 性能比率 ≥ 100%
- 🟡 接近基线: 80% ≤ 性能比率 < 100%
- 🟠 低于基线: 50% ≤ 性能比率 < 80%
- 🔴 远低于基线: 性能比率 < 50%

### 阶段 6: 性能优化

根据测试结果进行优化：

#### 6.1 分析瓶颈

```python
# 查看 test_results.json
{
  "performance": {
    "benchmarks": [
      {"shape": "batch_1", "M": 1, "N": 102400, "K": 5120,
       "latency_ms": 2.5, "gflops": 156.2}
    ],
    "baseline_comparison": {
      "performance_ratio": 15.3,  # 15.3% of baseline
      "current_gflops": 156.2,
      "baseline_gflops": 1021.5
    }
  }
}
```

#### 6.2 优化策略

**对于小批量 (M < 32)**：
- 使用 warp-level kernel (32 threads/block)
- 最小化 shared memory 使用
- 减少启动延迟

**对于大批量 (M ≥ 32)**：
- 使用 tiled kernel (16×128 blocks)
- 共享内存 tiling
- Double buffering

**内存受限场景**：
- 增加线程块大小
- 使用 shared memory 缓存权重
- 向量化内存加载 (float4, int4)

**计算受限场景**：
- Loop unrolling
- 减少 shared memory bank conflicts
- 使用 DP4A 指令

#### 6.3 迭代流程

```
┌─────────────────┐
│  生成初始版本   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  运行测试       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  分析结果       │
└────────┬────────┘
         ▼
   ┌─────┴─────┐
   ▼           ▼
编译失败     性能低
   │           │
   ▼           ▼
修复代码    优化代码
   │           │
   └─────┬─────┘
         ▼
┌─────────────────┐
│  重新测试       │
└────────┬────────┘
         │
         ▼
    满意? ──NO──┐
       │        │
      YES       └──> 继续优化
       │
       ▼
  ✅ 完成
```

## 常见问题诊断

### 编译错误

**错误：`no suitable constructor...__half`**
- **原因**：直接使用 `__half2float(block->d)` 转换 `uint16_t`
- **修复**：使用 union 方式转换
```cuda
__device__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}
// 使用: float d_w = read_half_as_float(w_block->d);
```

**错误：`blockDim.x * blockDim.y > 1024`**
- **原因**：线程块大小超过硬件限制
- **修复**：使用 `(32, 32)` 或 `(64, 16)` 等配置

**错误：缺少 `PYBIND11_MODULE`**
- **原因**：没有 PyTorch 绑定
- **修复**：在文件末尾添加：
```cuda
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Kernel forward");
}
```

### 正确性问题

**NMSE > 0.1**：
- 检查量化公式是否正确
- 确认 offset 补偿（Q4_0 需要 -8*s_a）
- 验证数据类型转换

### 性能问题

**远低于基线 (< 50%)**：
- 检查内存访问模式是否合并
- 增加线程块大小
- 使用 shared memory tiling

**接近基线 (80-100%)**：
- Loop unrolling
- 减少分支
- 使用向量化指令

---

## 性能分析

### 分析方法概览

| 方法 | 适用场景 | 优点 | 限制 |
|------|----------|------|------|
| **NCU Profiling** | Linux 原生环境 | 精确到指令级别的分析 | 需要 sudo 权限 |
| **简易 Benchmark 脚本** | 任意环境（含 WSL2） | 快速获取延迟/吞吐量 | 无法看到 stall 原因 |
| **torch.profiler** | 任意环境 | 不需 sudo，易集成 | 粒度较粗 |

---

### 方法 1: NVIDIA Nsight Compute (NCU) Profiling（推荐）

NCU 提供最精确的 kernel 级性能分析，可以识别内存瓶颈、Occupancy 限制、stall 原因等。

#### 1.1 环境要求

```bash
# 检查 ncu 是否可用
/usr/local/cuda-12.4/bin/ncu --version

# 检查是否需要 sudo（大部分服务器需要）
cat /proc/driver/nvidia/params 2>/dev/null | grep RmProfilingAdminOnly
# 输出 RmProfilingAdminOnly: 1 → 必须使用 sudo

# 检查 GPU 可用性（选择空闲的 GPU）
nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader
```

#### 1.2 两阶段流程

由于 `sudo` 环境下编译器路径可能丢失，推荐使用 **两阶段流程**：
1. **普通用户**：JIT 编译 kernel，生成 `.so` 缓存
2. **sudo + ncu**：加载预编译的 `.so`，执行 profiling

#### 1.3 创建 NCU Profiling 脚本

在 kernel 所在目录创建 `ncu_profile.py`：

```python
#!/usr/bin/env python3
"""NCU Profiling 脚本 - 加载预编译 .so，避免 sudo 下编译问题"""
import os, sys, argparse, importlib
import torch

def main():
    parser = argparse.ArgumentParser(description="NCU Profile - precompiled kernel")
    parser.add_argument("--M", type=int, default=1, help="Batch size")
    parser.add_argument("--N", type=int, default=5120, help="Output features")
    parser.add_argument("--K", type=int, default=5120, help="Input features")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat iterations")
    parser.add_argument("--so-path", type=str, required=True, help="Path to precompiled .so")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    print(f"📊 Profiling: M={M}, N={N}, K={K}", flush=True)

    # 直接加载预编译的 .so
    spec = importlib.util.spec_from_file_location("kernel_module", args.so_path)
    kernel_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_mod)
    kernel_func = kernel_mod.forward
    print(f"✅ 模块加载成功", flush=True)

    # 生成测试数据（根据 spec.json 中的格式调整）
    device = "cuda"
    num_blocks = K // 32
    bytes_per_block = 18  # Q4_0=18, Q8_0=34, Q8_1=36
    total_bytes = N * num_blocks * bytes_per_block
    weight_q = torch.randint(0, 256, (total_bytes,), dtype=torch.uint8, device=device)
    activation = torch.randn(M, K, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(args.warmup):
        _ = kernel_func(weight_q, activation, M, N, K)
    torch.cuda.synchronize()

    # Profile 运行
    for _ in range(args.repeat):
        output = kernel_func(weight_q, activation, M, N, K)
    torch.cuda.synchronize()
    print(f"✅ 完成! 输出形状: {output.shape}", flush=True)

if __name__ == "__main__":
    main()
```

#### 1.4 编译脚本（普通用户先编译）

在 kernel 目录创建 `profile_kernel.py`，用于首次 JIT 编译：

```python
#!/usr/bin/env python3
"""首次 JIT 编译 kernel，生成 .so 缓存供 ncu_profile.py 使用"""
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

script_dir = Path(__file__).parent
kernel_file = script_dir / "kernel.cu"

# 检测 GPU 架构
device_props = torch.cuda.get_device_properties(0)
major, minor = device_props.major, device_props.minor
print(f"🎯 GPU: {device_props.name}, Compute: {major}.{minor}")

gencode_flags = [
    f'-gencode=arch=compute_{major}{minor},code=compute_{major}{minor}',
    f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}',
]

# JIT 编译（.so 会缓存到 ~/.cache/torch_extensions/）
module = load(
    name="kernel_profile",
    sources=[str(kernel_file)],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'] + gencode_flags,
    verbose=False,
)
print(f"✅ 编译成功")

# 快速验证
M, N, K = 1, 5120, 5120  # 根据 spec.json 修改
device = "cuda"
num_blocks = K // 32
weight_q = torch.randint(0, 256, (N * num_blocks * 18,), dtype=torch.uint8, device=device)
activation = torch.randn(M, K, dtype=torch.float32, device=device)
output = module.forward(weight_q, activation, M, N, K)
print(f"✅ 验证通过，输出: {output.shape}")
```

#### 1.5 完整执行流程

```bash
# ========================================
# 步骤 1: 普通用户编译（生成 .so 缓存）
# ========================================
cd <kernel_directory>
CUDA_VISIBLE_DEVICES=1 python profile_kernel.py

# 查找编译后的 .so 文件
find ~/.cache/torch_extensions -name "kernel_profile.so" -type f
# 例如: ~/.cache/torch_extensions/py311_cu128/kernel_profile/kernel_profile.so

# ========================================
# 步骤 2: sudo ncu profiling
# ========================================
# 设置 .so 路径变量
SO_PATH=$(find ~/.cache/torch_extensions -name "kernel_profile.so" -type f | head -1)

# --- 输出 CSV 格式（终端查看） ---
sudo /usr/local/cuda-12.4/bin/ncu \
    --set full --csv \
    -k regex:<kernel_function_name> \
    env CUDA_VISIBLE_DEVICES=1 \
    /path/to/python ncu_profile.py \
    --M 1 --N 5120 --K 5120 \
    --so-path $SO_PATH

# --- 输出 .ncu-rep 文件（用 ncu-ui 可视化） ---
sudo /usr/local/cuda-12.4/bin/ncu \
    --set full \
    -k regex:<kernel_function_name> \
    -o ncu_report_M1 \
    env CUDA_VISIBLE_DEVICES=1 \
    /path/to/python ncu_profile.py \
    --M 1 --N 5120 --K 5120 \
    --so-path $SO_PATH
```

**关键参数说明**:

| 参数 | 说明 |
|------|------|
| `--set full` | 收集所有性能指标（推荐） |
| `--csv` | 输出 CSV 格式（适合终端和脚本解析） |
| `-k regex:<name>` | 只分析匹配名称的 kernel（过滤 PyTorch 内部 kernel） |
| `-o <filename>` | 输出 `.ncu-rep` 报告文件（可用 ncu-ui 打开） |
| `env CUDA_VISIBLE_DEVICES=1` | 指定 GPU（sudo 不允许直接设置环境变量） |

> **注意**: 使用 `sudo` 时不能用 `CUDA_VISIBLE_DEVICES=1 sudo ...` 的形式，
> 必须用 `sudo ... env CUDA_VISIBLE_DEVICES=1 ...` 的方式在 sudo 内部设置环境变量。

#### 1.6 NCU 关键指标解读

| 指标 | 含义 | 优秀 | 较差 | 优化方向 |
|------|------|------|------|----------|
| **Duration** | Kernel 执行时间 | 越小越好 | — | 整体优化 |
| **SM Busy** | SM 忙碌率 | >80% | <30% | 增加并行度 |
| **Memory Throughput** | 内存吞吐占峰值比 | >60% | <20% | 优化内存访问 |
| **Issued IPC** | 每周期发射指令数 | >2.0 | <1.0 | 减少 stall |
| **Achieved Occupancy** | 实际 Occupancy | >50% | <25% | 减少寄存器/shared mem |
| **Registers/Thread** | 每线程寄存器数 | <64 | >80 | 减少局部变量 |
| **L1 Hit Rate** | L1 缓存命中率 | >90% | <50% | 提高数据局部性 |
| **L2 Hit Rate** | L2 缓存命中率 | >50% | <15% | 数据预取 |
| **Scheduler Eligible** | 调度器有可发射 warp 的比率 | >60% | <30% | 增加活跃 warp |

#### 1.7 常见 NCU 警告与优化建议

| NCU 警告 | 原因 | 优化建议 |
|----------|------|----------|
| **UncoalescedGlobalAccess** | 全局内存访问未合并 | 让同一 warp 的线程访问连续地址 |
| **TheoreticalOccupancy (limited by registers)** | 寄存器数太多 | 使用 `__launch_bounds__`、减少局部变量、用 shared memory 替代 |
| **HighPipeUtilization (under-utilized)** | 计算管线利用不足 | 增加 Grid/Block 大小、减少 stall |
| **CPIStall (L1TEX)** | 等待 L1 纹理/全局内存 | 合并内存访问、使用 shared memory 缓存 |
| **CPIStall (barrier)** | 等待 `__syncthreads()` | 减少 barrier 次数、均衡 warp 工作负载 |
| **CPIStall (IMC miss)** | 常量内存缓存 miss | 减少常量访问、用寄存器/shared memory 替代 |

#### 1.8 NCU 分析实例

以下是一个 Q4_0 × FP32_INT8 warp kernel 在 A800 上的分析结果示例：

```
┌─────────────────────────────────────────────────────────────┐
│  NCU Profiling Report - w4a32c8_q4_0_kernel_warp (M=1)     │
│  GPU: NVIDIA A800 80GB PCIe, Compute 8.0                    │
├─────────────────────────────────────────────────────────────┤
│  Duration:              77.9 μs                              │
│  SM Busy:               52.0%        🟡 中等                │
│  Memory Throughput:     190 GB/s (83.9%)  🟡 内存受限       │
│  Max DRAM Bandwidth:    15.0%        🟡                     │
│  Issued IPC:            2.08         🟢 良好                │
│  FMA Pipeline:          49.2%        🟢 良好                │
│  L1 Hit Rate:           96.3%        🟢 优秀                │
│  L2 Hit Rate:           12.8%        🔴 很低                │
│  Registers/Thread:      80           🔴 偏高                │
│  Theoretical Occupancy: 37.5%        🔴 受寄存器限制        │
│  Achieved Occupancy:    35.3%        🔴 低                  │
│  Grid/Block:            640 × 256                            │
├─────────────────────────────────────────────────────────────┤
│  ⚠️ 主要瓶颈:                                               │
│  1. 非合并全局内存访问 (74% excessive sectors)               │
│  2. L1TEX 等待 (56% stall)                                  │
│  3. 寄存器压力 (80 regs → Occupancy 37.5%)                  │
└─────────────────────────────────────────────────────────────┘
```

**对应优化方案**:
- **P0**: 重构内存访问模式，确保同一 warp 内线程访问连续地址（减少 uncoalesced access）
- **P0**: 使用 `__launch_bounds__(256, 3)` 限制寄存器，提高 Occupancy
- **P1**: 将频繁访问的权重数据预取到 shared memory，减少 L1TEX stall
- **P2**: 增加数据局部性，提升 L2 Hit Rate

---

### 方法 2: PyTorch Profiler（推荐用于受限环境）

适用于 WSL2 环境、容器环境或不需要 sudo 权限的详细性能分析。

**重要**：在某些环境中（如 WSL2、容器环境、无管理员权限），无法使用 NCU (NVIDIA Compute Utility) 进行详细的性能分析。PyTorch Profiler 是最通用且功能强大的替代方案。

### PyTorch Profiler 详细分析（推荐）

当 NCU 不可用时（如 WSL2、容器环境），PyTorch 内置 Profiler 是最佳替代方案。

#### 完整 Profiler 脚本

项目根目录已提供 `profile_torch.py` 脚本，可直接使用。如果需要自定义，可参考以下模板：

```python
#!/usr/bin/env python3
"""PyTorch Profiler for CUDA Kernel Analysis"""
import torch
import sys
import os
import torch.profiler as profiler

# 设置 kernel 路径（根据实际实验目录调整）
KERNEL_DIR = "/root/Agent4Kernel/kernelevalplus/llm_kernel_test/sandbox/generated/<experiment_dir>/<variant>"
kernel_file = os.path.join(KERNEL_DIR, "kernel.cu")

# 解析命令行参数
M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

print(f"Profiling with M={M}, N={N}, K={K}")
print(f"Kernel type: {'warp' if M < 8 else 'tiled'}")

# JIT 编译
from torch.utils.cpp_extension import load

print("Compiling kernel...")
module = load(
    name=f"kernel_profile_m{M}",
    sources=[kernel_file],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-gencode=arch=compute_89,code=compute_89',
        '-gencode=arch=compute_89,code=sm_89',
    ],
    verbose=False
)
print("Compilation done!")

# 准备数据
num_blocks = K // 32
bytes_per_block = 18  # Q4_0 block size
total_weight_bytes = N * num_blocks * bytes_per_block

weight_q = torch.randint(0, 256, (total_weight_bytes,), dtype=torch.uint8, device='cuda')
activation = torch.randn(M, K, dtype=torch.float32, device='cuda')

# Warmup
print("Warming up...")
for _ in range(20):
    _ = module.forward(weight_q, activation, M, N, K)
torch.cuda.synchronize()

# 计算 FLOPs
total_flops = 2 * M * N * K  # multiply-add
print(f"Theoretical FLOPs: {total_flops:,}")

# ============== PyTorch Profiler 分析 ==============
print("\n" + "="*60)
print("Running PyTorch Profiler...")
print("="*60)

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    with_modules=True,
    on_trace_ready=profiler.tensorboard_trace_handler('./torch_profiler_logs'),
) as p:
    for _ in range(10):
        output = module.forward(weight_q, activation, M, N, K)
    torch.cuda.synchronize()

# 打印摘要
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# ============== 详细时序分析 ==============
print("\n" + "="*60)
print("Detailed Timing Analysis (CUDA Events)")
print("="*60)

iterations = 100
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 预热
for _ in range(10):
    _ = module.forward(weight_q, activation, M, N, K)
torch.cuda.synchronize()

# Benchmark
start_event.record()
for _ in range(iterations):
    output = module.forward(weight_q, activation, M, N, K)
end_event.record()
end_event.synchronize()

avg_time_ms = start_event.elapsed_time(end_event) / iterations
time_us = avg_time_ms * 1000

# 计算性能指标
flops = total_flops / (avg_time_ms / 1000)
gflops = flops / 1e9
tflops = flops / 1e12

# 内存带宽计算
weight_bytes = N * num_blocks * bytes_per_block
activation_bytes = M * K * 4
output_bytes = M * N * 4
total_io_bytes = weight_bytes + activation_bytes + output_bytes
bandwidth_gbps = total_io_bytes / (avg_time_ms / 1000) / 1e9

print(f"\nConfiguration: M={M}, N={N}, K={K}")
print(f"\nTiming:")
print(f"  Average latency: {avg_time_ms:.4f} ms ({time_us:.2f} us)")
print(f"\nCompute Performance:")
print(f"  GFLOPS: {gflops:.2f}")
print(f"  TFLOPS: {tflops:.4f}")
print(f"\nMemory Metrics:")
print(f"  Weight bytes:   {weight_bytes:,} ({weight_bytes/1e6:.2f} MB)")
print(f"  Activation bytes: {activation_bytes:,} ({activation_bytes/1e6:.2f} MB)")
print(f"  Output bytes:   {output_bytes:,} ({output_bytes/1e6:.2f} MB)")
print(f"  Total I/O:      {total_io_bytes:,} ({total_io_bytes/1e6:.2f} MB)")
print(f"  Bandwidth:      {bandwidth_gbps:.2f} GB/s")
print(f"\nCompute Intensity:")
arithmetic_intensity = total_flops / total_io_bytes
print(f"  Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")

# 瓶颈判断
print(f"\n" + "="*60)
print("Bottleneck Analysis")
print("="*60)

# RTX 4090 参数（根据实际 GPU 调整）
peak_bandwidth = 1008  # GB/s
peak_tflops_fp32 = 82.6

bw_util = (bandwidth_gbps / peak_bandwidth) * 100
compute_util = (gflops / 1000 / peak_tflops_fp32) * 100

print(f"  Bandwidth Utilization: {bw_util:.1f}% ({bandwidth_gbps:.1f} / {peak_bandwidth} GB/s)")
print(f"  Compute Utilization:   {compute_util:.1f}% ({gflops/1000:.2f} / {peak_tflops_fp32} TFLOPS)")

if arithmetic_intensity < 50:
    print(f"\n  Status: MEMORY BOUND (AI={arithmetic_intensity:.1f} < 50)")
    print(f"  Recommendation: 优化内存访问模式，增加数据重用")
elif arithmetic_intensity > 200:
    print(f"\n  Status: COMPUTE BOUND (AI={arithmetic_intensity:.1f} > 200)")
    print(f"  Recommendation: 优化计算效率，使用向量化指令")
else:
    print(f"\n  Status: BALANCED (AI={arithmetic_intensity:.1f})")

print("\n" + "="*60)
print("Profiler trace saved to ./torch_profiler_logs/")
print("View with: tensorboard --logdir=./torch_profiler_logs")
print("="*60)
```

#### 运行 Profiler

```bash
# 进入项目根目录
cd /root/Agent4Kernel/kernelevalplus

# 基本用法（需要先修改脚本中的 KERNEL_DIR 路径）
python profile_torch.py <M> <N> <K>

# 测试小 batch (warp kernel)
python profile_torch.py 1 4096 4096

# 测试大 batch (tiled kernel)
python profile_torch.py 32 4096 4096
python profile_torch.py 128 4096 4096
```

**注意**：使用前需要修改 `profile_torch.py` 中的 `KERNEL_DIR` 变量，指向实际的 kernel 目录。

#### 分析输出示例

```
============================================================
Detailed Timing Analysis (CUDA Events)
============================================================

Configuration: M=1, N=4096, K=4096
Kernel type: warp

Timing:
  Average latency: 0.0245 ms (24.54 us)

Compute Performance:
  GFLOPS: 1367.40
  TFLOPS: 1.3674

Memory Metrics:
  Weight bytes:   9,437,184 (9.44 MB)
  Activation bytes: 16,384 (0.02 MB)
  Output bytes:   16,384 (0.02 MB)
  Total I/O:      9,469,952 (9.47 MB)
  Bandwidth:      385.92 GB/s

Compute Intensity:
  Arithmetic Intensity: 3.54 FLOPs/byte

============================================================
Bottleneck Analysis
============================================================
  Bandwidth Utilization: 38.3% (385.9 / 1008 GB/s)
  Compute Utilization:   1.7% (1.37 / 82.6 TFLOPS)

  Status: MEMORY BOUND (AI=3.5 < 50)
  Recommendation: 优化内存访问模式，增加数据重用
```

#### TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir=./torch_profiler_logs

# 浏览器访问
# http://localhost:6006/#pytorch_profiler
```

TensorBoard 提供：
- **Overview**: 操作时间分布
- **Operator View**: 各操作的 CUDA 时间
- **Kernel View**: GPU kernel 详细信息
- **Trace View**: 时间线视图（最详细）

---

### 简单性能分析脚本

如果只需要快速测试性能，可以使用简化版本 `profile_simple.py`：

```python
# profile_simple.py
import torch
import sys
sys.path.insert(0, "/path/to/kernelevalplus")  # 修改为实际路径
from pathlib import Path
from llm_kernel_test.unified_test_runner import UnifiedTestRunner

runner = UnifiedTestRunner()
test_dir = Path(".")
runner._compile(test_dir)
module = runner._compiled_modules[str(test_dir)]

# 测试配置（从 spec.json 获取）
M, N, K = 1, 4096, 4096  # 根据实际情况修改
device = "cuda"

# 生成数据
num_blocks = K // 32
total_bytes = N * num_blocks * 18  # Q4_0: 18 bytes/block
weight_q = torch.randint(0, 256, (total_bytes,), dtype=torch.uint8, device=device)
activation = torch.randn(M, K, dtype=torch.float32, device=device)

# 预热
for _ in range(10):
    _ = module.forward(weight_q, activation, M, N, K)
torch.cuda.synchronize()

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
iterations = 1000
start.record()
for _ in range(iterations):
    output = module.forward(weight_q, activation, M, N, K)
end.record()
torch.cuda.synchronize()

avg_time_ms = start.elapsed_time(end) / iterations

# 计算性能指标
total_ops = 2 * M * N * K
throughput_gflops = (total_ops / (avg_time_ms / 1000)) / 1e9
total_bytes_transfer = total_bytes + M * K * 4 + M * N * 4
bandwidth_gb_s = total_bytes_transfer / (avg_time_ms / 1000) / 1e9

print(f"Latency: {avg_time_ms*1000:.3f} us")
print(f"Throughput: {throughput_gflops:.1f} GFLOPS")
print(f"Bandwidth: {bandwidth_gb_s:.1f} GB/s")

# 瓶颈分析
peak_tflops = 102  # 根据您的 GPU 调整（A800=312 TFLOPS FP16, A100=312, RTX 4090=82.6）
peak_bandwidth = 288  # 根据 GPU 调整（A800=2039 GB/s, A100=2039, RTX 4090=1008）
util_flops = (throughput_gflops / 1000) / peak_tflops * 100
util_bw = bandwidth_gb_s / peak_bandwidth * 100

print(f"\nCompute Efficiency: {util_flops:.1f}%")
print(f"Memory Efficiency: {util_bw:.1f}%")

if util_flops < util_bw * 1.2:
    print("Status: COMPUTE BOUND")
else:
    print("Status: MEMORY BOUND")
```

#### 2.2 运行

```bash
cd <experiment_directory>
python profile_simple.py
```

---

### 优化建议

根据性能分析结果，选择优化策略：

| 瓶颈类型 | 优化策略 |
|----------|----------|
| **COMPUTE BOUND** | Loop unrolling、向量化指令、减少分支 |
| **MEMORY BOUND** | Shared memory tiling、数据预取、向量化加载 |
| **低 Occupancy** | 使用 `__launch_bounds__` 限制寄存器、减少 shared memory 使用 |
| **非合并内存访问** | 重新设计线程-数据映射，确保同 warp 线程访问连续地址 |
| **Barrier Stall** | 减少 `__syncthreads()` 次数、均衡 warp 工作负载 |
| **Grid 太小** | 增加 tiling 粒度、改进多维度分 block 策略 |

---

### NCU Profiler 参考（权限可用时）

**⚠️ 注意**：NCU 需要 GPU 性能计数器访问权限，在以下环境中通常无法使用：
- WSL2 环境
- 容器环境（无特权模式）
- 无管理员权限的系统
- 某些云服务器环境

**如果无法使用 NCU，请使用 PyTorch Profiler（见上方）**，它提供了足够详细的性能分析。

当有 GPU 性能计数器访问权限时，NCU (Nsight Compute) 提供最详细的 kernel 分析。

#### 基本用法

```bash
# 完整 profile
ncu --set full -o report.ncu-rep python script.py

# 只分析特定 kernel
ncu --set full -k "kernel_name" -o report python script.py

# 分析子进程中的 kernel (PyTorch JIT 需要)
ncu --set full --target-processes all -o report /path/to/python script.py
```

#### 关键指标

| 指标类别 | 关键指标 | 说明 |
|----------|----------|------|
| **内存** | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` | 全局内存加载 sector 数 |
| **内存** | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | Shared memory bank conflicts |
| **计算** | `smsp__sass_thread_inst_executed_op_dfma_pred_on` | FP32 FMA 指令数 |
| **Occupancy** | `sm__warps_active.avg.pct_of_peak` | 活跃 warp 占比 |
| **吞吐** | `gpu__time_duration.sum` | GPU 执行时间 |

#### 分析命令示例

```bash
# 分析内存访问模式
ncu --metrics \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    -o mem_analysis python script.py

# 分析 shared memory bank conflicts
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    -o smem_analysis python script.py

# 分析计算吞吐
ncu --metrics \
    smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_dmul_pred_on.sum \
    -o compute_analysis python script.py
```

#### 查看 NCU 报告

```bash
# GUI 查看
ncu-ui report.ncu-rep

# 命令行摘要
ncu --import report.ncu-rep --page summary

# 导出为 CSV
ncu --import report.ncu-rep --csv > report.csv
```

---

## 最佳实践

### 1. 代码结构
- 清晰的函数划分
- 合理的注释
- 错误检查

### 2. 性能优化
- 先保证正确性，再优化性能
- 使用性能分析脚本识别瓶颈
- 逐步优化，每次验证

### 3. 测试验证
- 每次修改后重新测试
- 保持多个版本的代码用于对比
- 记录优化过程和结果

## 版本管理原则

### ⚠️ 重要：每次优化必须新建文件夹

**每次修改代码后，都必须创建新的文件夹保存，绝不能覆盖已有版本！**

### 目录结构规范

```
llm_kernel_test/sandbox/generated/
├── <definition_name>/                         # 主目录：使用定义文件的 name
│   │                                           # 例如: w4a32c8_q4_0_fp32_int8_deepseek_v2_lm_head_n102400_k5120
│   ├── attempt_<strategy>_<HHMMSS>/           # 单次尝试（带时间戳）
│   │   └── w4a32c8_q4_0_fp32_int8/
│   │       ├── kernel.cu                      # 当前版本代码
│   │       ├── spec.json                      # 规格说明
│   │       ├── reference.py                   # 参考实现
│   │       ├── test_results.json              # 测试结果
│   │       ├── optimization_log.md            # 优化日志（关键！）
│   │       └── metadata.json                  # 元数据
│   │
│   ├── attempt_<strategy>_<HHMMSS>/           # 下一个版本（新文件夹！）
│   │   └── w4a32c8_q4_0_fp32_int8/
│   │       ├── kernel.cu                      # 修改后的代码
│   │       └── ...
```

### 命名规范

| 类型 | 命名格式 | 示例 |
|------|----------|------|
| 主目录 | 使用定义文件的 `name` 字段 | `w4a32c8_q4_0_fp32_int8_deepseek_v2_lm_head_n102400_k5120` |
| Attempt 目录 | `attempt_<strategy>_<HHMMSS>` | `attempt_warp_142536` |
| 优化日志 | `optimization_log.md` | - |

### 完整示例

```
llm_kernel_test/sandbox/generated/
├── w4a32c8_q4_0_fp32_int8_deepseek_v2_lm_head_n102400_k5120/  # 主目录（定义文件的 name）
│   │
│   ├── attempt_basic_143022/                  # 14:30:22 - 基础版本
│   │   └── w4a32c8_q4_0_fp32_int8/
│   │       ├── kernel.cu                      # 简单实现
│   │       ├── test_results.json              # 45.2 GFLOPS (15%)
│   │       └── optimization_log.md            # 记录初始实现
│   │
│   ├── attempt_tiling_144815/                 # 14:48:15 - Tiling 优化
│   │   └── w4a32c8_q4_0_fp32_int8/
│   │       ├── kernel.cu                      # 添加 shared memory tiling
│   │       ├── test_results.json              # 156.8 GFLOPS (52%)
│   │       └── optimization_log.md
│   │           └── 内容：
│   │               - 基于 attempt_basic_143022
│   │               - 添加 128x128 shared memory tiling
│   │               - 性能提升: 45.2 → 156.8 GFLOPS (+247%)
│   │
│   ├── attempt_warp_150930/                   # 15:09:30 - Warp 优化
│   │   └── w4a32c8_q4_0_fp32_int8/
│   │       ├── kernel.cu                      # 小 M 使用 warp kernel
│   │       ├── test_results.json              # 287.3 GFLOPS (95%)
│   │       └── optimization_log.md
│   │           └── 内容：
│   │               - 基于 attempt_tiling_144815
│   │               - 添加 M<32 的 warp-level 分支
│   │               - 性能提升: 156.8 → 287.3 GFLOPS (+83%)
│   │
│   └── attempt_final_152147/                  # 15:21:47 - 最终版本
│       └── w4a32c8_q4_0_fp32_int8/
│           ├── kernel.cu                      # 所有优化整合
│           ├── test_results.json              # 312.5 GFLOPS (104%)
│           └── optimization_log.md
```

### 优化日志模板

每次创建新版本时，必须更新 `optimization_log.md`：

```markdown
# 优化日志 - <attempt_id>

## 基本信息
- **时间**: 2026-02-10 14:48:15
- **基于版本**: `attempt_basic_143022`
- **优化策略**: Shared Memory Tiling
- **目标性能**: >150 GFLOPS

## 修改内容

### 代码变更
- 添加 shared memory 声明: `__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE]`
- 实现数据预取到 shared memory
- 添加 double buffering 减少 bank conflict

### 性能对比
| 版本 | GFLOPS | Baseline % | 提升 |
|------|--------|------------|------|
| attempt_basic_143022 | 45.2 | 15% | - |
| attempt_tiling_144815 | 156.8 | 52% | +247% |

## 测试结果
- 编译: ✅ 成功
- 正确性: ✅ 通过 (NMSE: 0.000123)
- 性能: ✅ 显著提升

## 下一步计划
- 添加 warp-level 优化 (M<32 场景)
- 减少 shared memory bank conflicts

## 问题记录
- 无
```

### 使用 test_runner.py 的正确方式

```bash
# 定义文件的 name（从 JSON 文件中获取）
DEFINITION_NAME="w4a32c8_q4_0_fp32_int8_deepseek_v2_lm_head_n102400_k5120"
VARIANT="w4a32c8_q4_0_fp32_int8"

# ✅ 正确：每次使用新的时间戳
TIMESTAMP=$(date +%H%M%S)

# 提交代码
python llm_kernel_test/test_runner.py \
    --submit my_kernel.cu \
    --variant ${VARIANT} \
    --attempt-id ${DEFINITION_NAME}/attempt_warp_${TIMESTAMP}

# 运行测试
python llm_kernel_test/test_runner.py \
    --test \
    --variant ${VARIANT} \
    --attempt-id ${DEFINITION_NAME}/attempt_warp_${TIMESTAMP}

# ❌ 错误：覆盖已有版本
python llm_kernel_test/test_runner.py \
    --submit my_kernel_v2.cu \
    --variant ${VARIANT} \
    --attempt-id ${DEFINITION_NAME}/attempt_warp_144815  # 不要这样做！
```

**说明**：
- `DEFINITION_NAME`: 使用定义文件的 `name` 字段值
- `attempt_<strategy>_<HHMMSS>`: 每次新版本使用新的时间戳
- 最终路径: `llm_kernel_test/sandbox/generated/${DEFINITION_NAME}/attempt_<strategy>_<HHMMSS>/${VARIANT}/`

### 版本对比工具

可以创建脚本来对比同一问题下所有版本的性能：

```python
# compare_versions.py
import json
from pathlib import Path

def compare_versions(definition_name):
    """对比同一问题下所有版本的性能"""
    base_path = Path("llm_kernel_test/sandbox/generated") / definition_name
    results = {}
    for attempt_dir in Path(batch_dir).iterdir():
        if attempt_dir.is_dir():
            result_file = attempt_dir / "w4a32c8_q4_0_fp32_int8" / "test_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    results[attempt_dir.name] = {
                        "gflops": data["performance"]["benchmarks"][0]["gflops"],
                        "baseline": data["performance"]["baseline_comparison"]["performance_ratio"]
                    }

    # 打印对比表格
    print(f"{'Version':<30} {'GFLOPS':<15} {'Baseline %':<15}")
    print("-" * 60)
    for name, metrics in sorted(results.items()):
        print(f"{name:<30} {metrics['gflops']:<15.1f} {metrics['baseline']:<15.1f}")
```

## ⚠️ 绝对禁止的操作

1. **❌ 不要修改已保存的 kernel.cu** - 如需修改，创建新版本
2. **❌ 不要覆盖 test_results.json** - 它是版本的历史记录
3. **❌ 不要复用 attempt-id** - 每次使用新的时间戳
4. **❌ 不要删除旧版本** - 保留所有中间版本用于对比
