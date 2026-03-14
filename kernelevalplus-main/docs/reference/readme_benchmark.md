# KernelEvalPlus Benchmark System

一个完备的 CUDA Kernel 测试和性能评估框架，专注于量化 GEMM 算子。

## 🎯 核心特性

- ✅ **51 个问题定义** - 覆盖 LLaMA, DeepSeek-V2/V3, Qwen2.5 等主流模型
- ✅ **28 个算子实现** - 包含 W4A16, W8A8, W8A8C8 等多种量化方案
- ✅ **自动映射系统** - Definition 到 Operator 的智能关联
- ✅ **批量测试工具** - 一键测试所有算子
- ✅ **结构化报告** - HTML/Markdown/JSON 多格式输出
- ✅ **性能基准测试** - 集成 GFLOPS 计算和时间测量

## 🚀 快速开始

### 1. 环境准备

```bash
source /home/haiyan/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8
cd /home/haiyan/Agent4Kernel/KernelEvalPlus
```

### 2. 验证系统

```bash
./validate_benchmark.sh
```

### 3. 测试单个算子

```bash
cd python
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --benchmark
```

### 4. 批量测试

```bash
# 测试所有算子
python test_all.py --output results.json

# 只测试 DeepSeek-V3
python test_all.py --filter "deepseek_v3/*" --output results_ds3.json

# 生成 HTML 报告
python generate_report.py results.json --format html
```

## 📁 项目结构

```
KernelEvalPlus/
├── definitions/quant_gemm/          # 问题定义 (51 个 JSON)
│   ├── llama/                       # LLaMA 模型
│   ├── deepseek_v2/                 # DeepSeek-V2
│   ├── deepseek_v3/                 # DeepSeek-V3
│   ├── qwen2_5_7b/                  # Qwen2.5-7B
│   └── templates/                   # 模板定义
│
├── python/
│   ├── operators/quant_gemm/        # 算子实现 (28 个变体)
│   │   ├── variants/
│   │   │   ├── generic/             # 通用实现
│   │   │   ├── llama/               # LLaMA 优化
│   │   │   └── deepseek_v3/         # DeepSeek-V3 优化
│   │   └── ...
│   │
│   ├── quant_gemm/                  # 核心模块
│   │   ├── _C.so                    # CUDA 扩展
│   │   ├── __init__.py              # Python 接口
│   │   └── csrc/                    # C++/CUDA 源码
│   │
│   ├── test_operator.py             # 单算子测试
│   ├── test_operator_enhanced.py    # 增强版 (JSON 输出)
│   ├── test_all.py                  # 批量测试
│   ├── generate_mapping.py          # 映射生成
│   └── generate_report.py           # 报告生成
│
├── definition_operator_mapping.json # 映射文件
├── BENCHMARK_GUIDE.md               # 详细文档
├── IMPROVEMENTS_SUMMARY.md          # 改进总结
└── validate_benchmark.sh            # 验证脚本
```

## 🔧 核心工具

### 1. generate_mapping.py - 映射生成器

自动扫描并建立 definition 和 operator 的映射关系。

```bash
python python/generate_mapping.py
```

**输出：**
```
✅ Mapping saved to: definition_operator_mapping.json
Statistics:
  Total definitions: 38
  Total operators: 28
  Mapped: 37
  Unmapped: 1
```

### 2. test_operator.py - 单算子测试

测试单个算子的正确性和性能。

```bash
python test_operator.py <name> <path> [options]

# 选项：
#   --benchmark          运行性能测试
#   --config "M=1,N=512,K=512"  自定义配置
#   --iterations 100     Benchmark 迭代次数
```

### 3. test_all.py - 批量测试

批量测试所有或部分算子。

```bash
python test_all.py [options]

# 选项：
#   --filter "pattern"   过滤测试 (如 "deepseek_v3/*")
#   --limit N            限制测试数量
#   --benchmark          包含性能测试
#   --output file.json   保存结果
#   --dry-run            预览测试计划
```

### 4. generate_report.py - 报告生成

从 JSON 结果生成可视化报告。

```bash
python generate_report.py results.json --format html|markdown

# 选项：
#   --format html        生成 HTML 报告
#   --format markdown    生成 Markdown 报告
#   --output file        自定义输出文件
```

## 📊 测试结果示例

### 正确性测试

```
============================================================
 Testing: w8a8c8_q8_0_q8_1
============================================================
Folder: operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1

------------------------------------------------------------
 Correctness Tests
------------------------------------------------------------
[PASS] custom_0: nmse=7.1288e-15 (threshold=0.05)
[PASS] custom_1: nmse=7.5846e-15 (threshold=0.05)
[PASS] custom_2: nmse=1.0773e-14 (threshold=0.05)

Results: 3 passed, 0 failed
```

### 性能测试

```
------------------------------------------------------------
 Benchmarks
------------------------------------------------------------
custom_0             M=    1 N=  512 K=  512 |    0.027 ms |    19.32 GFLOPS
custom_1             M=    8 N= 1024 K= 1024 |    0.066 ms |   254.83 GFLOPS
```

## 📖 文档

- **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** - 完整使用指南
- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - 架构改进总结
- **[TEST_OPERATOR_GUIDE.md](TEST_OPERATOR_GUIDE.md)** - 测试工具详解

## 🎓 支持的量化方案

| 变体 | 权重 | 激活 | 计算 | 模型 |
|------|------|------|------|------|
| W4A16 | Q4_0 | FP16 | FP16 | LLaMA |
| W4A32C32 | Q4_0 | FP32 | FP32 | DeepSeek-V3 |
| W8A16 | Q8_0 | FP16 | FP16 | LLaMA |
| W8A32C8 | Q8_0 | FP32 | INT8 | DeepSeek-V3 |
| W8A8C8 | Q8_0 | Q8_1 | INT8 | DeepSeek-V3 |
| W4A8 | Q4_0 | Q8_1 | INT8 | Generic |

## 🔬 精度指标

- **NMSE (Normalized Mean Squared Error)** - 默认指标
- **阈值：** 0.05 (可配置)
- **实测精度：** 10^-14 ~ 10^-15 (接近浮点精度极限)

## 🏆 性能数据

### W8A8C8 Q8_0×Q8_1 (DeepSeek-V3)

| 配置 | 时间 (ms) | GFLOPS | 精度 (NMSE) |
|------|-----------|--------|-------------|
| M=1, N=512, K=512 | 0.027 | 19.32 | 5.79e-15 |
| M=8, N=1024, K=1024 | 0.066 | 254.83 | 1.01e-14 |

## 🛠️ 开发指南

### 添加新的 Operator

1. 在 `operators/quant_gemm/variants/` 创建目录
2. 添加必需文件：
   - `spec.json` - 算子规格
   - `reference.py` - Python 参考实现
   - `kernel.cu` - CUDA kernel
   - `bindings.cpp` - Python 绑定
3. 重新生成映射：`python generate_mapping.py`
4. 运行测试验证

### 添加新的 Definition

1. 在 `definitions/quant_gemm/` 对应模型目录添加 JSON
2. 遵循 schema 规范（参考现有文件）
3. 重新生成映射：`python generate_mapping.py`
4. 运行测试验证

## 🐛 故障排查

### 测试失败

```bash
# 启用详细输出
python test_operator_enhanced.py ... --verbose

# 启用 CUDA 同步模式
export CUDA_LAUNCH_BLOCKING=1
python test_operator.py ...
```

### 性能问题

```bash
# 使用小尺寸快速验证
python test_operator.py ... --config "M=1,N=32,K=32"

# 限制批量测试数量
python test_all.py --limit 5
```

## 📈 统计数据

- **问题定义：** 51 个 (38 个非模板)
- **算子实现：** 28 个变体
- **映射覆盖：** 97.4% (37/38)
- **支持模型：** 4 个 (LLaMA, DeepSeek-V2/V3, Qwen2.5)
- **量化格式：** 8 种 (Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0, Q8_1, Q4_K)

## 🤝 贡献

欢迎贡献新的算子实现、问题定义或工具改进！

## 📝 许可证

[根据项目实际情况填写]

## 🙏 致谢

- llama.cpp - 量化格式参考
- PyTorch - 深度学习框架
- CUDA - GPU 加速计算

---

**最后更新：** 2026-02-03
**版本：** 1.0
**状态：** ✅ 生产就绪
