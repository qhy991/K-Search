# LLM Kernel 测试框架

用于测试 LLM 生成的 CUDA kernel 代码的自动化框架。

## 📁 目录结构

```
llm_kernel_test/
├── sandbox/                    # 沙箱环境（隔离测试）
│   ├── generated/              # LLM 生成的代码
│   │   ├── attempt_1/
│   │   ├── attempt_2/
│   │   └── ...
│   └── build/                  # 编译输出
│
├── templates/                  # 模板文件（从主项目复制）
│   ├── bindings.cpp.template
│   ├── spec.json.template
│   └── reference.py.template
│
├── baseline/                   # 基线代码（人工实现的参考）
│   └── w4a8_q4_0_q8_1/
│       └── kernel.cu
│
├── test_runner.py              # 测试运行器
├── evaluator.py                # 评估器
├── report_generator.py         # 报告生成器
└── results/                    # 测试结果
    ├── attempt_1_results.json
    ├── attempt_2_results.json
    └── comparison_report.md
```

## 🚀 工作流程

### 1. 准备阶段

```bash
# 创建测试环境
python llm_kernel_test/test_runner.py --setup --variant w4a8_q4_0_q8_1
```

这会：
- 创建沙箱目录
- 复制必要的模板文件
- 准备测试配置

### 2. 提交 LLM 生成的代码

```bash
# 提交新的 kernel 代码
python llm_kernel_test/test_runner.py \
    --submit generated_kernel.cu \
    --variant w4a8_q4_0_q8_1 \
    --attempt-id attempt_1
```

这会：
- 将代码复制到 `sandbox/generated/attempt_1/`
- 自动填充模板文件
- 准备编译环境

### 3. 自动测试

```bash
# 运行完整测试
python llm_kernel_test/test_runner.py \
    --test \
    --attempt-id attempt_1 \
    --variant w4a8_q4_0_q8_1
```

这会：
1. **编译检查**：尝试编译代码
2. **正确性测试**：运行测试用例
3. **性能测试**：运行 benchmark
4. **对比基线**：与人工实现对比

### 4. 查看结果

```bash
# 生成报告
python llm_kernel_test/report_generator.py \
    --compare attempt_1 attempt_2 baseline
```

## 📊 测试指标

### 编译检查
- ✅ 编译成功
- ❌ 编译失败（记录错误信息）

### 正确性测试
- NMSE（归一化均方误差）
- 通过率（24 个测试用例）

### 性能测试
- 延迟（ms）
- 吞吐量（GFLOPS）
- 与基线对比（加速比）

### 代码质量
- 代码行数
- 是否使用预定义函数
- 内存访问模式

## 🔧 配置文件

`test_config.json`:
```json
{
  "sandbox_dir": "llm_kernel_test/sandbox",
  "templates_dir": "llm_kernel_test/templates",
  "baseline_dir": "llm_kernel_test/baseline",
  "results_dir": "llm_kernel_test/results",

  "test_shapes": [
    {"M": 1, "N": 4096, "K": 4096},
    {"M": 8, "N": 4096, "K": 4096},
    {"M": 32, "N": 8192, "K": 8192}
  ],

  "benchmark_config": {
    "warmup": 10,
    "iterations": 100
  },

  "correctness_threshold": {
    "nmse": 0.1
  }
}
```

## 📝 使用示例

### 完整流程

```bash
# 1. 设置测试环境
python llm_kernel_test/test_runner.py --setup --variant w4a8_q4_0_q8_1

# 2. 提交 LLM 生成的代码（第一次尝试）
python llm_kernel_test/test_runner.py \
    --submit llm_output_v1.cu \
    --variant w4a8_q4_0_q8_1 \
    --attempt-id v1

# 3. 运行测试
python llm_kernel_test/test_runner.py --test --attempt-id v1

# 4. 查看结果
cat llm_kernel_test/results/v1_results.json

# 5. 如果失败，修改 prompt，重新生成
python llm_kernel_test/test_runner.py \
    --submit llm_output_v2.cu \
    --variant w4a8_q4_0_q8_1 \
    --attempt-id v2

# 6. 对比多个版本
python llm_kernel_test/report_generator.py --compare v1 v2 baseline
```

## 🎯 测试报告示例

```markdown
# LLM Kernel 测试报告

## 测试概览

| Attempt | 编译 | 正确性 | 性能 (GFLOPS) | 相对基线 |
|---------|------|--------|---------------|----------|
| v1      | ✅   | ❌ (NMSE=0.5) | N/A | N/A |
| v2      | ✅   | ✅ (NMSE=0.01) | 85.2 | 0.87x |
| v3      | ✅   | ✅ (NMSE=0.008) | 98.4 | 1.00x |
| baseline| ✅   | ✅ (NMSE=0.005) | 98.1 | 1.00x |

## 详细分析

### v1 - 失败
- **编译**: 成功
- **正确性**: 失败（NMSE=0.5，超过阈值 0.1）
- **问题**: 未正确处理 Q4_0 的解包

### v2 - 部分成功
- **编译**: 成功
- **正确性**: 通过（NMSE=0.01）
- **性能**: 85.2 GFLOPS（比基线慢 13%）
- **问题**: 未使用 shared memory 优化

### v3 - 成功 ✅
- **编译**: 成功
- **正确性**: 通过（NMSE=0.008）
- **性能**: 98.4 GFLOPS（与基线相当）
- **优点**: 使用了 shared memory，性能优秀

## 推荐

**推荐使用**: v3
**原因**: 正确性和性能都达到要求
```

## 🔄 持续集成

可以集成到 CI/CD 流程：

```yaml
# .github/workflows/test_llm_kernel.yml
name: Test LLM Generated Kernel

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup environment
        run: |
          python llm_kernel_test/test_runner.py --setup

      - name: Test kernel
        run: |
          python llm_kernel_test/test_runner.py --test --attempt-id ${{ github.sha }}

      - name: Generate report
        run: |
          python llm_kernel_test/report_generator.py --attempt-id ${{ github.sha }}
```

## 📚 相关文档

- [测试运行器文档](test_runner.md)
- [评估器文档](evaluator.md)
- [报告生成器文档](report_generator.md)
