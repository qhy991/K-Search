# LLM API 配置完整指南

## 概述

KernelEvalPlus 现在支持灵活的 LLM API 配置，可以方便地切换不同的 LLM 提供商、模型和 API endpoint。

## 配置优先级（从高到低）

1. **命令行参数** - 临时覆盖，适合测试
2. **环境变量** - 推荐，安全且灵活
3. **配置文件** - `llm_kernel_test/test_config.json`
4. **默认值** - 最后的兜底配置

## 方法 1: 环境变量（推荐）

### 设置环境变量

```bash
# Linux/Mac
export DEEPSEEK_API_KEY="sk-xxx"
export OPENAI_API_KEY="sk-xxx"
export ANTHROPIC_API_KEY="sk-ant-xxx"

# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-xxx"
$env:OPENAI_API_KEY="sk-xxx"
$env:ANTHROPIC_API_KEY="sk-ant-xxx"
```

### 使用示例

```bash
# 设置 DeepSeek API Key
export DEEPSEEK_API_KEY="sk-xxx"

# 运行生成（自动使用环境变量中的 key）
python -m python.tools.llm_generator \
    definitions/quant_gemm/llama/w8a32c8_q8_0_q8_1_llama3_8b_att_qkv_n12288_k4096.json \
    --provider deepseek
```

## 方法 2: 命令行参数

### 完整命令示例

```bash
# DeepSeek (推荐 - 国内访问快)
python -m python.tools.llm_generator \
    definitions/quant_gemm/llama/w8a32c8_q8_0_q8_1_llama3_8b_att_qkv_n12288_k4096.json \
    --provider deepseek \
    --model deepseek-v3 \
    --api-key sk-xxx

# OpenAI GPT-4
python -m python.tools.llm_generator \
    definitions/quant_gemm/qwen3/w8a32c8_q8_0_q8_1_qwen3_32b_ffn_up_n25600_k5120.json \
    --provider openai \
    --model gpt-4o \
    --api-key sk-xxx \
    --base-url https://api.openai.com/v1

# Anthropic Claude
python -m python.tools.llm_generator \
    definitions/... \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --api-key sk-ant-xxx

# 使用代理/自定义端点
python -m python.tools.llm_generator \
    definitions/... \
    --provider openai \
    --model gpt-4o \
    --api-key sk-xxx \
    --base-url https://your-proxy.com/v1
```

### 可用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--provider` | LLM 提供商 | `deepseek`, `openai`, `anthropic` |
| `--model` | 模型名称 | `deepseek-v3`, `gpt-4o`, `claude-sonnet-4-20250514` |
| `--api-key` | API Key | `sk-xxx` |
| `--base-url` | API Base URL | `https://api.openai.com/v1` |
| `--prompt-style` | Prompt 风格 | `full`, `minimal`, `focused` |
| `--skip-test` | 跳过测试 | 只生成代码，不运行测试 |
| `--dry-run` | 只生成 prompt | 不调用 LLM API |

## 方法 3: 配置文件

编辑 `llm_kernel_test/test_config.json`:

```json
{
  "llm": {
    "active_provider": "deepseek",
    "providers": {
      "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "timeout": 300
      },
      "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-20250514",
        "timeout": 300
      },
      "deepseek": {
        "base_url": "https://cloud.infini-ai.com/maas/v1",
        "default_model": "deepseek-v3",
        "timeout": 300
      }
    }
  }
}
```

## 常用 LLM 提供商

### 1. DeepSeek (推荐 - 国内访问快)

```bash
export DEEPSEEK_API_KEY=sk-xxx
python -m python.tools.llm_generator \
    definitions/... \
    --provider deepseek \
    --model deepseek-v3
```

**优点**:
- 国内访问速度快
- 价格便宜
- 支持 deepseek-v3 模型（性能接近 GPT-4）

### 2. OpenAI

```bash
export OPENAI_API_KEY=sk-xxx
python -m python.tools.llm_generator \
    definitions/... \
    --provider openai \
    --model gpt-4o
```

### 3. Anthropic Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
python -m python.tools.llm_generator \
    definitions/... \
    --provider anthropic \
    --model claude-sonnet-4-20250514
```

## 批量生成示例

```bash
# 设置 API Key
export DEEPSEEK_API_KEY=sk-xxx

# 批量生成所有 qwen3 定义
python -m python.tools.llm_generator \
    --all \
    --pattern qwen3 \
    --provider deepseek

# 批量生成所有定义，跳过测试（只生成代码）
python -m python.tools.llm_generator \
    --all \
    --skip-test
```

## 安全建议

1. ✅ **使用环境变量** - 不要在命令行中传递 `--api-key`（会留在 shell 历史）
2. ✅ **使用 .env 文件** - 添加到 `.gitignore`
3. ✅ **使用密钥管理服务** - 生产环境推荐
4. ❌ **不要在代码中硬编码** API Key
5. ❌ **不要提交 .env 文件** 到版本控制

## .env 文件示例

```bash
# .env (添加到 .gitignore)
DEEPSEEK_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
```

## 快速开始

```bash
# 1. 设置 API Key
export DEEPSEEK_API_KEY=sk-your-key-here

# 2. 生成单个定义的 kernel
python -m python.tools.llm_generator \
    definitions/quant_gemm/llama/w8a32c8_q8_0_q8_1_llama3_8b_att_qkv_n12288_k4096.json \
    --provider deepseek

# 3. 测试生成的 kernel
python llm_kernel_test/batch_test_runner.py \
    --test-all \
    --batch-dir llm_kernel_test/sandbox/generated
```
