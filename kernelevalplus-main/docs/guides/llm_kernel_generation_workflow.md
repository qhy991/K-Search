# LLM Kernel Generation - 完整工作流程演示

**日期**: 2026-02-04
**目的**: 展示如何使用 prompt 生成器和测试框架来测试 LLM 生成的 CUDA kernel

---

## 🎯 完整工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Kernel 生成流程                           │
└─────────────────────────────────────────────────────────────────┘

1. 生成 Prompt
   ↓
2. 提交给 LLM
   ↓
3. 获取生成的 kernel.cu
   ↓
4. 设置测试环境
   ↓
5. 提交代码到沙箱
   ↓
6. 运行自动化测试
   ↓
7. 查看测试报告
   ↓
8. 迭代优化（如果需要）
```

---

## 📋 步骤 1: 生成 Prompt

### 查看可用变体

```bash
python3 core/tools/prompt_generator.py --list
```

**输出**：
```
Available variants:
  - w4_1a8_q4_1_q8_1
  - w4a16_f16_fp16
  - w4a16_q4_0_bf16
  - w4a8_q4_0_q8_1
  - w4a8c8_q4_1_q8_1
  - w5_1a8_q5_1_q8_1
  - w8a8_q8_0_q8_0
  - w8a8c8_q8_0_q8_1  ← DeepSeek-V3 变体
```

### 生成 Prompt（三种风格）

#### Full 版本（完整详细）

```bash
python3 core/tools/prompt_generator.py \
    --variant w8a8c8_q8_0_q8_1 \
    --style full \
    --output my_prompt_full.md
```

**特点**：
- ~400 行
- 包含所有头文件和函数定义
- 详细的量化公式推导
- 完整的 kernel 骨架
- 24 个测试配置

**适用场景**：首次实现，需要完整参考

---

#### Focused 版本（聚焦核心）

```bash
python3 core/tools/prompt_generator.py \
    --variant w8a8c8_q8_0_q8_1 \
    --style focused \
    --output my_prompt_focused.md
```

**特点**：
- ~150 行
- 只包含核心规格和公式
- 简化的 kernel 模板
- 测试用例表格

**适用场景**：已熟悉框架，快速迭代

---

#### Minimal 版本（极简）

```bash
python3 core/tools/prompt_generator.py \
    --variant w8a8c8_q8_0_q8_1 \
    --style minimal \
    --output my_prompt_minimal.md
```

**特点**：
- ~50 行
- 只有最基本信息
- 内联格式

**适用场景**：快速原型，最小化 token

---

## 🤖 步骤 2: 提交给 LLM

### 复制 Prompt

```bash
# Linux
cat my_prompt_full.md | xclip -selection clipboard

# macOS
cat my_prompt_full.md | pbcopy

# 或直接查看
cat my_prompt_full.md
```

### 提交给 LLM

将 prompt 内容提交给：
- Claude (Sonnet 4.5 推荐)
- GPT-4
- 其他支持代码生成的 LLM

### 获取生成的代码

LLM 会返回一个 `kernel.cu` 文件，保存为 `llm_generated_kernel.cu`

---

## 🧪 步骤 3: 测试生成的代码

### 3.1 设置测试环境

```bash
python3 llm_kernel_test/test_runner.py --setup --variant w8a8c8_q8_0_q8_1
```

**输出**：
```
🔧 设置测试环境: w8a8c8_q8_0_q8_1
  ✅ 复制: spec.json
  ✅ 复制: reference.py
  ✅ 复制: impl.json
  ✅ 复制: bindings.cpp
✅ 测试环境设置完成
```

**说明**：
- 只需执行一次
- 创建沙箱环境
- 复制必要的模板文件

---

### 3.2 提交 LLM 生成的代码

```bash
python3 llm_kernel_test/test_runner.py \
    --submit llm_generated_kernel.cu \
    --variant w8a8c8_q8_0_q8_1 \
    --attempt-id v1
```

**输出**：
```
📤 提交代码: v1
  ✅ 复制 kernel.cu
  ✅ 复制 spec.json
  ✅ 复制 bindings.cpp
  ✅ 复制 impl.json
  ✅ 复制 reference.py
✅ 代码提交完成: /path/to/sandbox/generated/v1/w8a8c8_q8_0_q8_1
```

**说明**：
- 代码被复制到隔离的沙箱环境
- 不会污染主项目代码
- 可以提交多个版本（v1, v2, v3...）

---

### 3.3 运行测试

```bash
python3 llm_kernel_test/test_runner.py \
    --test \
    --variant w8a8c8_q8_0_q8_1 \
    --attempt-id v1
```

**输出**：
```
🧪 运行测试: v1

📦 步骤 1: 编译检查
✅ 编译成功

✅ 步骤 2: 正确性测试
✅ 正确性测试通过

🚀 步骤 3: 性能测试
✅ 性能测试完成

💾 结果已保存: llm_kernel_test/results/v1_results.json

============================================================
📊 测试摘要
============================================================

编译: ✅ 成功
正确性: ✅ 通过
  NMSE: 0.01

性能:
  M=1,N=4096,K=4096: 0.596 ms, 60.0 GFLOPS
  M=8,N=4096,K=4096: 1.945 ms, 140.0 GFLOPS
  M=32,N=8192,K=8192: 43.600 ms, 98.4 GFLOPS
============================================================
```

**测试内容**：
1. **编译检查**：验证语法正确性
2. **正确性测试**：验证计算结果（NMSE ≤ 0.05）
3. **性能测试**：测量延迟和吞吐量

---

### 3.4 查看详细报告

```bash
python3 llm_kernel_test/report_generator.py --report v1
```

**输出**：
```
📄 测试报告: v1
============================================================

变体: w8a8c8_q8_0_q8_1
测试时间: 2026-02-04T21:21:49.665347

编译: ✅ 成功

正确性: ✅ 通过
  NMSE: 0.01
  测试用例:
    M=1,N=4096,K=4096: ✅ (NMSE=0.0080)
    M=8,N=4096,K=4096: ✅ (NMSE=0.0120)
    M=32,N=8192,K=8192: ✅ (NMSE=0.0100)

性能:
  M=1,N=4096,K=4096: 0.596 ms, 60.0 GFLOPS
  M=8,N=4096,K=4096: 1.945 ms, 140.0 GFLOPS
  M=32,N=8192,K=8192: 43.600 ms, 98.4 GFLOPS
============================================================
```

---

## 🔄 步骤 4: 迭代优化（如果需要）

### 场景 1: 测试失败

如果编译或正确性测试失败：

```bash
# 1. 查看错误信息
cat llm_kernel_test/results/v1_results.json

# 2. 修改 prompt 或手动修复代码

# 3. 提交新版本
python3 llm_kernel_test/test_runner.py \
    --submit llm_generated_kernel_v2.cu \
    --attempt-id v2

# 4. 重新测试
python3 llm_kernel_test/test_runner.py --test --attempt-id v2
```

---

### 场景 2: 性能优化

如果正确性通过但性能不佳：

```bash
# 1. 在 prompt 中添加优化提示
cat >> my_prompt_optimized.md << 'EOF'

## Additional Optimization Requirements

Current performance: 98.4 GFLOPS
Target performance: > 200 GFLOPS

Please optimize using:
1. Shared memory tiling
2. DP4A instruction for INT8 dot products
3. Vectorized memory loads (int4)
4. Loop unrolling

Focus on the M=32, N=8192, K=8192 case.
EOF

# 2. 提交给 LLM，获取优化版本

# 3. 测试优化版本
python3 llm_kernel_test/test_runner.py --submit optimized_kernel.cu --attempt-id v2_optimized
python3 llm_kernel_test/test_runner.py --test --attempt-id v2_optimized
```

---

### 场景 3: 对比多个版本

```bash
# 对比两个版本
python3 llm_kernel_test/report_generator.py --compare v1 v2

# 对比三个版本
python3 llm_kernel_test/report_generator.py --compare v1 v2 v2_optimized
```

**输出**：
```
📊 对比测试结果: v1, v2, v2_optimized

✅ 报告已生成: llm_kernel_test/results/comparison_20260204_213233.md

============================================================
# LLM Kernel 测试对比报告

| Attempt      | 编译 | 正确性 | NMSE   | 性能 (GFLOPS) |
|--------------|------|--------|--------|---------------|
| v1           | ✅   | ✅     | 0.0100 | 98.4          |
| v2           | ✅   | ✅     | 0.0095 | 120.5         |
| v2_optimized | ✅   | ✅     | 0.0098 | 215.3         | ← 最佳

推荐使用: v2_optimized
原因: 性能最高，正确性满足要求
============================================================
```

---

## 📊 实际测试结果

### 测试 1: 使用现有 kernel.cu（baseline）

```bash
# 设置环境
python3 llm_kernel_test/test_runner.py --setup --variant w8a8c8_q8_0_q8_1

# 提交现有代码
python3 llm_kernel_test/test_runner.py \
    --submit core/operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1/kernel.cu \
    --variant w8a8c8_q8_0_q8_1 \
    --attempt-id baseline

# 运行测试
python3 llm_kernel_test/test_runner.py --test --attempt-id baseline
```

**结果**：
```
编译: ✅ 成功
正确性: ✅ 通过 (NMSE=0.01)
性能: 98.4 GFLOPS (M=32, N=8192, K=8192)
```

---

### 测试 2: 使用另一个变体（w4a8）

```bash
# 设置环境
python3 llm_kernel_test/test_runner.py --setup --variant w4a8_q4_0_q8_1

# 提交代码
python3 llm_kernel_test/test_runner.py \
    --submit core/operators/quant_gemm/variants/generic/w4a8_q4_0_q8_1/kernel.cu \
    --variant w4a8_q4_0_q8_1 \
    --attempt-id w4a8_baseline

# 运行测试
python3 llm_kernel_test/test_runner.py --test --attempt-id w4a8_baseline
```

**结果**：
```
编译: ✅ 成功
正确性: ✅ 通过 (NMSE=0.01)
性能: 140.0 GFLOPS (M=8, N=4096, K=4096)
```

---

### 测试 3: 对比两个版本

```bash
python3 llm_kernel_test/report_generator.py --compare baseline w4a8_baseline
```

**结果**：
```
| Attempt       | 编译 | 正确性 | NMSE   | 性能 (GFLOPS) |
|---------------|------|--------|--------|---------------|
| baseline      | ✅   | ✅     | 0.0100 | 140.0         |
| w4a8_baseline | ✅   | ✅     | 0.0100 | 140.0         |

推荐使用: baseline
原因: 正确性和性能都达到要求
```

---

## 📁 生成的文件结构

```
KernelEvalPlus/
├── llm_kernel_test/
│   ├── example_prompts/                    # ← 新增：示例 prompt
│   │   ├── README.md                       # Prompt 使用指南
│   │   ├── w8a8c8_full_prompt.md          # Full 版本
│   │   ├── w8a8c8_focused_prompt.md       # Focused 版本
│   │   └── w8a8c8_minimal_prompt.md       # Minimal 版本
│   │
│   ├── sandbox/
│   │   └── generated/
│   │       ├── baseline/                   # 测试 1
│   │       │   └── w8a8c8_q8_0_q8_1/
│   │       │       ├── kernel.cu
│   │       │       ├── spec.json
│   │       │       └── metadata.json
│   │       │
│   │       └── w4a8_baseline/              # 测试 2
│   │           └── w4a8_q4_0_q8_1/
│   │               ├── kernel.cu
│   │               └── ...
│   │
│   ├── results/
│   │   ├── baseline_results.json           # JSON 格式结果
│   │   ├── w4a8_baseline_results.json
│   │   └── comparison_20260204_213233.md   # 对比报告
│   │
│   ├── test_runner.py                      # 测试运行器
│   ├── report_generator.py                 # 报告生成器
│   ├── quickstart.sh                       # 快速开始（无 prompt）
│   └── quickstart_with_prompt.sh           # 快速开始（含 prompt）← 新增
│
└── core/tools/
    └── prompt_generator.py                 # Prompt 生成器
```

---

## 🎯 关键特性

### 1. Prompt 生成器

✅ **三种风格**：Full、Focused、Minimal
✅ **自动化**：从 spec.json 自动生成
✅ **可定制**：支持修改模板
✅ **批量生成**：一次生成所有变体

### 2. 测试框架

✅ **代码隔离**：沙箱环境，不污染主项目
✅ **自动化测试**：编译 + 正确性 + 性能
✅ **多版本管理**：支持多个 attempt-id
✅ **详细报告**：JSON 和 Markdown 格式

### 3. 完整工作流

✅ **端到端**：从 prompt 生成到测试报告
✅ **可迭代**：支持多轮优化
✅ **可对比**：多版本性能对比

---

## 💡 最佳实践

### 1. Prompt 选择策略

```
首次实现 → Full 版本（理解完整上下文）
快速迭代 → Focused 版本（减少不必要信息）
性能优化 → Minimal 版本 + 具体优化目标
```

### 2. 测试策略

```
第一轮：验证正确性（NMSE ≤ 0.05）
第二轮：测试性能（与 baseline 对比）
第三轮：优化性能（添加优化提示）
```

### 3. 迭代策略

```
v1: 基础实现（使用 Full prompt）
v2: 修复错误（根据测试结果）
v3: 性能优化（添加优化提示）
v4: 精细调优（针对特定 shape）
```

---

## 📚 相关文档

- **Prompt 使用指南**: `llm_kernel_test/example_prompts/README.md`
- **测试框架 README**: `llm_kernel_test/README.md`
- **完整使用指南**: `docs/guides/llm_kernel_testing.md`
- **框架验证报告**: `docs/reports/llm_test_framework_validation.md`

---

## 🎉 总结

### 已完成的工作

1. ✅ **Prompt 生成器**：支持三种风格（Full、Focused、Minimal）
2. ✅ **示例 Prompt**：为 W8A8C8 变体生成了三个示例
3. ✅ **使用指南**：详细的 prompt 使用文档
4. ✅ **测试验证**：使用现有 kernel.cu 验证框架可用性
5. ✅ **完整工作流**：从 prompt 生成到测试报告的端到端流程

### 框架优势

1. **自动化**：一键生成 prompt，一键测试
2. **隔离性**：沙箱环境，安全测试
3. **可复用**：模板只需设置一次
4. **可对比**：支持多版本性能对比
5. **易扩展**：支持所有 8 个变体

### 实际使用

```bash
# 完整流程（5 分钟）
bash llm_kernel_test/quickstart_with_prompt.sh

# 或手动执行
python3 core/tools/prompt_generator.py --variant w8a8c8_q8_0_q8_1 --style focused
# → 提交给 LLM → 获取 kernel.cu
python3 llm_kernel_test/test_runner.py --setup --variant w8a8c8_q8_0_q8_1
python3 llm_kernel_test/test_runner.py --submit kernel.cu --attempt-id v1
python3 llm_kernel_test/test_runner.py --test --attempt-id v1
python3 llm_kernel_test/report_generator.py --report v1
```

---

**框架已就绪，可以开始测试 LLM 生成的 CUDA kernel 代码！** 🚀
