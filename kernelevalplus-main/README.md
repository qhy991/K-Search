# KernelEvalPlus

CUDA Kernel Benchmark 框架，用于评估 LLM 生成 CUDA kernel 的能力。

## 项目简介

KernelEvalPlus 是一个 CUDA kernel benchmark 框架，提供：

- **算子定义**: 标准化的算子规格定义 (JSON 格式)
- **测试框架**: 编译、正确性验证、性能测试
- **Baseline 数据**: 各 GPU 的性能基线数据
- **自动化流程**: 通过 Claude Code 完成 kernel 开发与测试

### 主要用途

1. **Benchmark 测试**: 评估 LLM 生成 CUDA kernel 的能力
2. **框架验证**: 验证测试程序、编译流程、性能评估是否完备
3. **性能对比**: 与 baseline 数据对比，评估 kernel 质量

## 目录结构

```
kernelevalplus/
├── definitions/               # 算子定义 (JSON)
│   ├── flash_attention/      # Flash Attention 定义
│   ├── quant_gemm/           # 量化 GEMM 定义
│   ├── rms_norm/             # RMS Norm 定义
│   └── topk/                 # TopK 定义
│
├── core/                     # 核心模块
│   ├── operators/            # 算子框架
│   └── tools/                # 工具脚本
│       ├── baseline_api.py   # Baseline 查询 API
│       └── gpu_specs.py      # GPU 规格数据
│
├── llm_kernel_test/          # 测试执行层
│   ├── compiler/             # JIT 编译器
│   │   ├── jit_compiler.py   # PyTorch JIT 编译
│   │   └── error_analyzer.py # 编译错误分析
│   ├── reference/            # 参考实现
│   │   ├── quantize.py       # 量化函数
│   │   └── gemm_ref.py       # GEMM 参考实现
│   ├── op_test_handler.py    # 算子测试处理器
│   └── unified_test_runner.py # 统一测试运行器
│
├── include/                  # CUDA 头文件
│   ├── quant_types.h         # 量化类型定义 (Q4_0, Q8_0, Q8_1)
│   ├── gemm_cuda_dp4a.cuh    # DP4A 优化 GEMM
│   └── quantize.h            # 量化/反量化函数
│
├── kernels/                  # CUDA kernel 参考实现
│
├── config/                   # 配置文件
├── data/baseline/            # Baseline 性能数据
│
├── output/                   # 实验输出目录
│
└── batch_claude_parallel.py  # 批量处理脚本
```

## Benchmark 流程

```
┌─────────────────────────────────────────────────────────────┐
│                    KernelEvalPlus (Benchmark)               │
│                                                             │
│  提供资源:                                                   │
│  • definitions/*.json  → 算子规格定义                        │
│  • data/baseline/      → 性能基线数据                        │
│  • llm_kernel_test/    → 编译、测试、验证框架                 │
│  • include/            → 量化类型、参考实现                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code (被测对象)                    │
│                                                             │
│  执行任务:                                                   │
│  1. 读取 definition，理解算子规格                            │
│  2. 生成 CUDA kernel 代码                                    │
│  3. 编译、测试、验证正确性                                    │
│  4. 性能优化迭代                                             │
│  5. 对比 baseline，输出结果                                   │
└─────────────────────────────────────────────────────────────┘
```

## 支持的算子

### quant_gemm (量化矩阵乘法)

| 变体 | 权重 | 激活 | 性能指标 |
|------|------|------|----------|
| w4a32c8 | Q4_0/Q4_1 | FP32/INT8 | TFLOPS |
| w8a32c8 | Q8_0 | FP32/INT8 | TFLOPS |
| w8a8c8 | Q8_0 | Q8_1 | TFLOPS |

### flash_attention

| 变体 | 缓存类型 | 性能指标 |
|------|----------|----------|
| f16 | FP16 | TFLOPS |
| q4_0 | Q4_0 量化 | TFLOPS |
| q8_0 | Q8_0 量化 | TFLOPS |

### rms_norm

| 参数 | 性能指标 |
|------|----------|
| hidden_size | GB/s |

### topk

| 参数 | 性能指标 |
|------|----------|
| k, vocab_size | GB/s |

## Baseline 数据查询

```python
from core.tools.baseline_api import get_baseline_api

api = get_baseline_api()

# GEMM baseline
api.get_gemm("RTX4090", "q4_0", M=4096, N=1, K=4096)
# 返回: {"tflops": 0.85, "us_per_run": 123.4}

# Flash Attention baseline
api.get_flash_attn("RTX4090", "Llama3-8B", "F16", seq_len=512, batch=512)

# RMS Norm baseline
api.get_rms_norm("RTX4090", hidden_size=4096, ne=[4096, 512, 1, 1])

# TopK baseline
api.get_topk("RTX4090", k=8, vocab_size=256, ne=[256, 512, 1, 1])
```

## 算子定义格式

```json
{
  "name": "w4a32c8_q4_0_fp32_int8_llama3_8b",
  "op_type": "quant_gemm",
  "axes": {
    "M": {"type": "var", "values": [1, 32, 512]},
    "N": {"type": "const", "value": 4096},
    "K": {"type": "const", "value": 4096}
  },
  "inputs": {
    "weight": {"dtype": "block_q4_0", "shape": ["N", "K/32"]},
    "activation": {"dtype": "float32", "shape": ["M", "K"]}
  },
  "outputs": {
    "output": {"dtype": "float32", "shape": ["M", "N"]}
  }
}
```

## 量化格式

### Q4_0 (18 bytes / 32 values)

```
+----------------+--------------------------------+
| d (2 bytes)    |        qs[16] (16 bytes)       |
| half scale     | packed 4-bit values            |
+----------------+--------------------------------+
```

### Q8_0 (34 bytes / 32 values)

```
+----------------+--------------------------------+
| d (2 bytes)    |        qs[32] (32 bytes)       |
| half scale     | 8-bit signed values            |
+----------------+--------------------------------+
```

### Q8_1 (36 bytes / 32 values)

```
+------------------+--------------------------------+
| ds (4 bytes)     |        qs[32] (32 bytes)       |
| half2 (d + sum)  | 8-bit signed values            |
+------------------+--------------------------------+
```

## 使用方法

### 单任务测试

```bash
python llm_kernel_test/unified_test_runner.py --test \
    --definition definitions/quant_gemm/llama/w4a32c8_q4_0_llama3_8b.json \
    --attempt-path output/glm_5/quant_gemm/task/attempts/v1
```

**为什么传入目录而非 kernel 文件路径？**

测试程序需要目录结构来管理多个文件：

```
attempt_dir/                 # 传入此目录路径
├── kernel.cu               # 必需：kernel 源码
├── spec.json               # 可选：算子规格（没有则用 --definition）
├── auto_wrapper.cu         # 自动生成：PyTorch wrapper
└── test_results.json       # 自动生成：测试结果
```

这种设计的好处：
1. **结果保存**：测试结果写回同目录，方便追踪
2. **自动生成**：编译时自动生成 wrapper 文件
3. **完整记录**：一次尝试的所有文件集中管理

### 批量测试 (通过 Claude Code)

```bash
python batch_claude_parallel.py definitions/quant_gemm \
    --model glm-5 \
    -w 3
```

### 测试流程

```
unified_test_runner.py 测试流程:

┌─────────────────────────────────────────────────────────────┐
│ 输入: attempt_dir/ (包含 kernel.cu)                         │
│       --definition xxx.json (算子规格)                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 编译                                                │
│   • 读取 kernel.cu                                          │
│   • 检测 GPU 架构，设置 nvcc flags                           │
│   • 自动生成 PyTorch wrapper (如需要)                        │
│   • JIT 编译                                                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: 正确性验证                                          │
│   • 生成测试数据                                            │
│   • 运行 kernel                                             │
│   • 计算参考输出                                            │
│   • 计算 NMSE，检查 NaN/Inf                                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: 性能测试                                            │
│   • Warmup (10次) + Benchmark (100次)                       │
│   • 计算 TFLOPS 或 GB/s                                     │
│   • 对比 baseline 数据                                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 输出: attempt_dir/test_results.json                         │
└─────────────────────────────────────────────────────────────┘
```

## 测试输出

```json
{
  "task_name": "w4a32c8_q4_0_llama3_8b",
  "success": true,
  "correctness": {
    "nmse": 1.2e-6,
    "passed": true
  },
  "performance": {
    "tflops": 0.85,
    "baseline_tflops": 0.92,
    "speedup": 0.92
  },
  "hardware": "RTX4090",
  "timestamp": "2026-03-14T12:00:00"
}
```

## 许可证

本项目仅供研究和开发使用。
