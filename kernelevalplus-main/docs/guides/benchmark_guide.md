# KernelEvalPlus Benchmark 完整使用指南

## 架构概览

```
KernelEvalPlus/
├── definitions/                    # 问题定义 (51 个 JSON 文件)
├── python/
│   ├── operators/                  # 算子实现 (28 个变体)
│   ├── test_operator.py            # 单个算子测试
│   ├── test_operator_enhanced.py   # 增强版测试 (支持 JSON)
│   ├── test_all.py                 # 批量测试工具
│   ├── generate_mapping.py         # 生成 definition-operator 映射
│   └── generate_report.py          # 生成测试报告
└── definition_operator_mapping.json # Definition-Operator 映射文件
```

## 快速开始

### 1. 生成 Definition-Operator 映射

首次使用前，需要生成映射文件：

```bash
cd /home/haiyan/Agent4Kernel/KernelEvalPlus
source /home/haiyan/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

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

### 2. 测试单个算子

#### 基础测试
```bash
cd python
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1
```

#### 带性能测试
```bash
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --benchmark --iterations 100
```

#### 自定义配置
```bash
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --config "M=1,N=512,K=512" \
    --config "M=8,N=1024,K=1024"
```

#### JSON 输出（增强版）
```bash
python test_operator_enhanced.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --output-json --output-file results.json
```

### 3. 批量测试

#### 测试所有算子
```bash
cd python
python test_all.py --output results_all.json
```

#### 测试特定模型
```bash
# 只测试 DeepSeek-V3
python test_all.py --filter "deepseek_v3/*" --output results_ds3.json

# 只测试 LLaMA
python test_all.py --filter "llama/*" --output results_llama.json

# 只测试 W8A8C8 变体
python test_all.py --filter "*w8a8c8*" --output results_w8a8c8.json
```

#### 带性能测试
```bash
python test_all.py --benchmark --output results_with_bench.json
```

#### 限制测试数量（快速验证）
```bash
python test_all.py --limit 5 --output results_quick.json
```

#### 预览测试计划（不实际运行）
```bash
python test_all.py --filter "deepseek_v3/*" --dry-run
```

### 4. 生成测试报告

#### HTML 报告
```bash
python generate_report.py results_all.json --format html
# 生成: results_all.html
```

#### Markdown 报告
```bash
python generate_report.py results_all.json --format markdown
# 生成: results_all.md
```

#### 自定义输出文件
```bash
python generate_report.py results_all.json --format html --output report.html
```

## 完整工作流示例

### 场景 1: 验证新实现的 kernel

```bash
# 1. 测试单个 operator
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --benchmark

# 2. 如果通过，测试所有相关 definitions
python test_all.py --filter "*w8a8c8_q8_0_q8_1*" --output w8a8c8_results.json

# 3. 生成报告
python generate_report.py w8a8c8_results.json --format html
```

### 场景 2: 回归测试

```bash
# 1. 运行完整测试套件
python test_all.py --benchmark --output baseline.json

# 2. 修改代码后重新测试
python test_all.py --benchmark --output current.json

# 3. 对比结果（手动或使用 diff 工具）
diff baseline.json current.json
```

### 场景 3: 性能分析

```bash
# 1. 只测试性能关键的配置
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --config "M=1,N=7168,K=7168" \
    --config "M=16,N=7168,K=7168" \
    --config "M=128,N=7168,K=7168" \
    --benchmark --iterations 1000

# 2. 分析不同尺寸的性能特征
```

## 文件说明

### 1. definition_operator_mapping.json

映射文件结构：
```json
{
  "version": "1.0",
  "statistics": {
    "total_definitions": 38,
    "total_operators": 28,
    "mapped_definitions": 37,
    "unmapped_definitions": 1
  },
  "mappings": [
    {
      "definition": "definitions/quant_gemm/deepseek_v3/w8a8c8_q8_0_q8_1_ds3_att_out_n7168_k7168.json",
      "operator": "core/operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1",
      "variant": "w8a8c8_q8_0_q8_1",
      "model": "deepseek-v3"
    }
  ]
}
```

### 2. 测试结果 JSON

```json
{
  "timestamp": "2026-02-03 20:45:00",
  "total_tests": 37,
  "summary": {
    "passed": 35,
    "failed": 2,
    "timeout": 0,
    "error": 0
  },
  "results": [
    {
      "definition": "definitions/...",
      "operator": "operators/...",
      "status": "passed",
      "duration": 1.23,
      "tests": [...]
    }
  ]
}
```

## 常见问题

### Q1: 如何添加新的 operator？

1. 在 `operators/quant_gemm/variants/` 创建新目录
2. 添加 `spec.json`, `reference.py`, `kernel.cu`
3. 重新生成映射：`python generate_mapping.py`

### Q2: 如何添加新的 definition？

1. 在 `definitions/quant_gemm/` 对应模型目录添加 JSON
2. 重新生成映射：`python generate_mapping.py`
3. 运行测试验证

### Q3: 测试失败如何调试？

```bash
# 1. 使用 verbose 模式
python test_operator_enhanced.py ... --verbose

# 2. 启用 CUDA 同步模式
export CUDA_LAUNCH_BLOCKING=1
python test_operator.py ...

# 3. 查看详细错误
python test_operator_enhanced.py ... --output-json | jq '.tests[].error'
```

### Q4: 如何跳过慢速测试？

```bash
# 使用 --limit 限制测试数量
python test_all.py --limit 10

# 或只测试小尺寸配置
python test_operator.py ... --config "M=1,N=32,K=32"
```

## 性能优化建议

### 1. 并行测试（未来改进）

当前 `test_all.py` 是串行的，可以改进为并行：
```python
# 使用 multiprocessing 或 concurrent.futures
from concurrent.futures import ProcessPoolExecutor
```

### 2. 缓存 reference 结果

对于相同配置，可以缓存 reference 计算结果。

### 3. 使用更快的 reference

考虑用 NumPy 向量化或 Numba JIT 加速 reference 实现。

## 总结

你的 benchmark 架构现在具备：

✅ **完整的三层结构**
- 问题定义 (definitions/)
- 参考实现 (reference.py)
- 测试程序 (test_operator.py, test_all.py)

✅ **自动化映射系统**
- generate_mapping.py 自动建立 definition-operator 关联

✅ **批量测试能力**
- test_all.py 支持过滤、限制、benchmark

✅ **结构化输出**
- JSON 格式结果
- HTML/Markdown 报告

✅ **灵活的配置**
- 自定义测试参数
- 模式匹配过滤
- 性能测试集成

**下一步改进方向：**
1. 添加回归检测（对比历史结果）
2. CI/CD 集成脚本
3. 性能趋势图表
4. 并行测试支持
