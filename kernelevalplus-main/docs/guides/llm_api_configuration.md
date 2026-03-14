# LLM API 配置示例

## 方法 1: 环境变量 (推荐用于生产环境)

```bash
# OpenAI
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o"

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-xxx"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"

# DeepSeek
export DEEPSEEK_API_KEY="sk-xxx"
export DEEPSEEK_BASE_URL="https://cloud.infini-ai.com/maas/v1"
export DEEPSEEK_MODEL="deepseek-v3"

# 然后直接运行
python -m python.tools.llm_generator definitions/...
```

## 方法 2: 命令行参数 (推荐用于临时测试)

```bash
# OpenAI
python -m python.tools.llm_generator \
    definitions/quant_gemm/llama/w8a32c8_q8_0_q8_1_llama3_8b_att_qkv_n12288_k4096.json \
    --provider openai \
    --model gpt-4o \
    --api-key sk-xxx

# Anthropic Claude
python -m python.tools.llm_generator \
    definitions/... \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --api-key sk-ant-xxx

# DeepSeek (国内，推荐)
python -m python.tools.llm_generator \
    definitions/... \
    --provider deepseek \
    --model deepseek-v3 \
    --api-key sk-xxx

# 自定义 Base URL (例如使用代理)
python -m python.tools.llm_generator \
    definitions/... \
    --provider openai \
    --model gpt-4o \
    --api-key sk-xxx \
    --base-url https://your-proxy.com/v1
```

## 方法 3: 配置文件 (llm_kernel_test/test_config.json)

```json
{
  "llm": {
    "active_provider": "deepseek",
    "providers": {
      "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "timeout": 300,
        "temperature": 0.0,
        "max_tokens": 8192
      },
      "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-20250514",
        "timeout": 300,
        "temperature": 0.0,
        "max_tokens": 8192
      },
      "deepseek": {
        "base_url": "https://cloud.infini-ai.com/maas/v1",
        "default_model": "deepseek-v3",
        "timeout": 300,
        "temperature": 0.0,
        "max_tokens": 8192
      }
    }
  }
}
```

## 配置优先级 (从高到低)

1. **命令行参数** `--api-key`, `--base-url`, `--model`
2. **环境变量** `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, etc.
3. **配置文件** `test_config.json`
4. **默认值**

## 常用 LLM 提供商配置

### DeepSeek (推荐 - 国内访问快)
```bash
--provider deepseek
--model deepseek-v3
--api-key sk-xxx
--base-url https://cloud.infini-ai.com/maas/v1
```

### OpenAI
```bash
--provider openai
--model gpt-4o
--api-key sk-xxx
--base-url https://api.openai.com/v1
```

### Anthropic Claude
```bash
--provider anthropic
--model claude-sonnet-4-20250514
--api-key sk-ant-xxx
--base-url https://api.anthropic.com
```

### 使用代理/自定义端点
```bash
--provider openai
--model gpt-4o
--api-key sk-xxx
--base-url https://your-proxy.com/openai/v1
```

## 批量生成示例

```bash
# 生成所有 qwen3 定义，使用 DeepSeek
export DEEPSEEK_API_KEY=sk-xxx
python -m python.tools.llm_generator --all --pattern qwen3 --provider deepseek

# 生成所有定义，跳过测试（仅生成代码）
python -m python.tools.llm_generator --all --skip-test
```

## 安全建议

1. **不要在代码中硬编码 API Key**
2. **使用环境变量存储敏感信息**
3. **将 `.env` 文件添加到 `.gitignore`**
4. **对于生产环境，考虑使用密钥管理服务**
