# 批量测试方案分析

## 当前状态分析

### 1. 现有资源

**模板变体** (`llm_kernel_test/templates/`):
```
w4a8_q4_0_q8_1/    # W4A8 量化，Q4_0×Q8_1
w8a32c8_q8_0_q8_1/ # W8A32C8 量化，Q8_0×Q8_1 (DeepSeek-V3)
w8a8c8_q8_0_q8_1/  # W8A8C8 量化，Q8_0×Q8_1
```

**已生成的尝试** (`llm_kernel_test/sandbox/generated/`):
```
deepseek_auto_test/     # 首次尝试（有占位符代码）
deepseek_v3_final/      # ✅ 成功版本（24/24 通过）
deepseek_v3_complete/   # 完整代码版本
deepseek_v3_fixed_prompt/ # 修复 prompt 后的版本
deepseek_v3_with_struct/ # 包含结构体定义的版本
... (还有其他)
```

**测试结果** (`llm_kernel_test/results/`):
```
deepseek_v3_final_results.json  # ✅ 成功
deepseek_v3_complete_results.json
deepseek_v3_with_struct_results.json
...
```

### 2. 当前测试框架

**单次测试命令**:
```bash
python llm_kernel_test/test_runner.py \
    --test \
    --variant w8a32c8_q8_0_q8_1 \
    --attempt-id deepseek_v3_final
```

**限制**:
- 只能单个测试
- 没有批量处理
- 没有汇总报告

## 批量测试需求

### 需求 1: 测试所有已生成的尝试
对 `sandbox/generated/` 下的所有 attempt_id 运行测试

### 需求 2: 测试所有模板变体
对所有模板变体运行测试（需要先生成代码）

### 需求 3: 生成汇总报告
- 总体通过率
- 编译成功率
- 性能对比
- 错误分类

## 实现方案

### 方案 A: 简单批量脚本

```bash
#!/bin/bash
# 批量测试所有已生成的尝试

for attempt_dir in llm_kernel_test/sandbox/generated/*/; do
    attempt_id=$(basename "$attempt_dir")
    for variant_dir in "$attempt_dir"/*; do
        variant=$(basename "$variant_dir")
        echo "Testing: $attempt_id / $variant"

        python llm_kernel_test/test_runner.py \
            --test \
            --variant "$variant" \
            --attempt-id "$attempt_id"
    done
done
```

### 方案 B: 扩展 test_runner.py

添加 `--batch-test` 命令：

```python
parser.add_argument("--batch-test", action="store_true",
                   help="批量测试所有已生成的尝试")
parser.add_argument("--batch-variants", nargs="*", default=[],
                   help="指定要测试的变体列表")
```

### 方案 C: 独立批量测试工具

创建 `batch_test_runner.py`：

```python
class BatchTestRunner:
    def test_all_attempts(self, pattern="*"):
        """测试所有匹配的 attempt"""

    def test_all_variants(self, variants=[]):
        """测试指定变体的所有 attempt"""

    def generate_report(self, output="batch_test_report.md"):
        """生成汇总报告"""
```

## 推荐方案

**方案 B + C 组合**:
1. 在 `test_runner.py` 添加简单的批量测试命令
2. 创建独立的 `batch_test_runner.py` 用于复杂批量测试

## 批量测试功能规划

### 1. 快速批量测试
```bash
# 测试所有已生成的尝试
python llm_kernel_test/test_runner.py --batch-test

# 只测试特定的 attempt
python llm_kernel_test/test_runner.py --batch-test --attempts deepseek_v3_final,deepseek_v3_complete
```

### 2. 按变体批量测试
```bash
# 测试特定变体的所有 attempt
python llm_kernel_test/test_runner.py --batch-test-variants --variants w8a32c8_q8_0_q8_1
```

### 3. 汇总报告
```bash
python llm_kernel_test/batch_test_runner.py --report --output batch_report.md
```

## 测试报告格式

```markdown
# 批量测试报告

## 总体统计
- 总测试数: 50
- 编译成功: 45 (90%)
- 正确性通过: 40 (89% of compiled)
- 平均性能: 100 GFLOPS

## 按变体分类
### w8a32c8_q8_0_q8_1 (DeepSeek-V3)
- deepseek_v3_final: ✅ 通过 (70-285 GFLOPS)
- deepseek_v3_complete: ❌ 编译失败
- deepseek_v3_with_struct: ❌ 编译失败

### w8a8c8_q8_0_q8_1
- (测试结果)

## 错误分类
- 编译错误: 5
  - 结构体未定义: 3
  - 符号不匹配: 2
- 运行时错误: 0

## 性能排行
1. deepseek_v3_final: 285 GFLOPS (small_batch)
2. ...
```
