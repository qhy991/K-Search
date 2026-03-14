# llama.cpp 对 DeepSeek 模型的支持

## 支持情况

**llama.cpp 完全支持 DeepSeek 模型**，包括三个版本：

1. **DeepSeek（原始版本）** - `LLM_ARCH_DEEPSEEK`
2. **DeepSeek V2** - `LLM_ARCH_DEEPSEEK2`
3. **DeepSeek V3** - 通过 `deepseek-v3` 标识和 chat template 支持

## 架构定义

### 枚举定义

**文件**：`src/llama-arch.h`

```cpp
enum llm_arch {
    // ... 其他架构 ...
    LLM_ARCH_DEEPSEEK,      // DeepSeek 原始版本
    LLM_ARCH_DEEPSEEK2,     // DeepSeek V2
    // ...
};
```

### 架构名称映射

**文件**：`src/llama-arch.cpp`

```cpp
static const std::map<llm_arch, std::string> LLM_ARCH_NAMES = {
    // ...
    { LLM_ARCH_DEEPSEEK,         "deepseek"         },
    { LLM_ARCH_DEEPSEEK2,        "deepseek2"        },
    // ...
};
```

## 模型实现

### DeepSeek（原始版本）

**文件**：`src/models/deepseek.cpp`

- **类名**：`llm_build_deepseek`
- **架构标识**：`LLM_ARCH_DEEPSEEK`
- **特点**：
  - 标准的 Transformer 架构
  - 支持 RoPE（旋转位置编码）
  - 支持注意力机制和 FFN

**关键代码结构**：
```cpp
llm_build_deepseek::llm_build_deepseek(const llama_model & model, 
                                       const llm_graph_params & params) {
    // 构建计算图
    // 1. 输入嵌入
    inpL = build_inp_embd(model.tok_embd);
    
    // 2. 位置编码
    inp_pos = build_inp_pos();
    
    // 3. 多层 Transformer
    for (int il = 0; il < n_layer; ++il) {
        // 注意力层
        // FFN 层
    }
}
```

### DeepSeek V2

**文件**：`src/models/deepseek2.cpp`

- **类名**：`llm_build_deepseek2`
- **架构标识**：`LLM_ARCH_DEEPSEEK2`
- **特点**：
  - **MoE（Mixture of Experts）架构**
  - **MLA（Multi-head Latent Attention）支持**
  - **YaRN RoPE 扩展**（支持长上下文）
  - **KV LoRA 支持**

**关键代码结构**：
```cpp
llm_build_deepseek2::llm_build_deepseek2(const llama_model & model, 
                                         const llm_graph_params & params) {
    const bool is_mla = hparams.is_mla();
    
    // MLA 特定的头大小计算
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();
    
    // YaRN RoPE 预缩放（关键优化）
    const float mscale = attn_factor_org * 
        (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));
    
    // MoE 专家路由
    // ...
}
```

**DeepSeek V2 特殊功能**：

1. **MoE 支持**：
   - 多个专家（experts）
   - 专家路由（expert routing）
   - 激活的专家数量可配置

2. **MLA（Multi-head Latent Attention）**：
   - 压缩的注意力机制
   - 减少 KV cache 大小
   - 提高推理效率

3. **YaRN RoPE**：
   - 长上下文支持
   - 动态频率缩放
   - 预缩放优化（避免重复计算）

## 模型注册

### 模型构建器注册

**文件**：`src/llama-model.cpp`

```cpp
static std::unique_ptr<llm_graph_context> llama_model_build_graph(
    const llama_model & model, const llm_graph_params & params) {
    
    switch (model.arch) {
        // ...
        case LLM_ARCH_DEEPSEEK:
            return std::make_unique<llm_build_deepseek>(model, params);
            
        case LLM_ARCH_DEEPSEEK2:
            return std::make_unique<llm_build_deepseek2>(model, params);
        // ...
    }
}
```

## 量化支持

llama.cpp 对 DeepSeek 模型支持所有标准量化格式：

- **Q4_0, Q4_1** - 4-bit 量化
- **Q5_0, Q5_1** - 5-bit 量化
- **Q8_0, Q8_1** - 8-bit 量化
- **Q2_K, Q3_K, Q4_K, Q5_K, Q6_K** - K-quants
- **FP16, BF16** - 半精度

**关键点**：DeepSeek V2 的 MoE 架构在量化时，每个专家独立量化。

## Tokenizer 支持

**文件**：`tests/CMakeLists.txt`

```cmake
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder 
    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
    
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm 
    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
```

支持两种 tokenizer：
- **deepseek-coder** - 代码专用
- **deepseek-llm** - 通用语言模型

## Chat Template 支持

**文件**：`tools/server/README.md`

llama.cpp 支持 DeepSeek 的聊天模板：

```bash
--chat-template deepseek    # DeepSeek 原始版本
--chat-template deepseek2   # DeepSeek V2
--chat-template deepseek3   # DeepSeek V3（如果支持）
```

## Reasoning Format 支持

**文件**：`tools/server/server-task.cpp`

```cpp
if (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY) {
    params.chat_parser_params.reasoning_in_content = params.stream && ...;
}
```

支持 DeepSeek R1 的推理格式：
- `--reasoning-format deepseek` - 标准格式
- `--reasoning-format deepseek-legacy` - 遗留格式

## 与定义文件的对应关系

### DeepSeek V2 定义文件

你的定义文件中的 DeepSeek V2 场景：

```json
{
  "model_architectures": ["deepseek-v2"],
  "name": "w4a16_q4_0_fp32_ds2_att_qkv_n15360_k5120",
  "description": "DeepSeek-V2 Attention QKV projection"
}
```

**对应 llama.cpp**：
- 架构：`LLM_ARCH_DEEPSEEK2`
- 计算图：`llm_build_deepseek2`
- 量化：支持 Q4_0 等格式
- MoE：支持多专家路由

### DeepSeek V3 定义文件

你的定义文件中的 DeepSeek V3 场景：

```json
{
  "model_architectures": ["deepseek-v3"],
  "name": "w4a16_q4_0_fp32_ds3_att_qkv_n21504_k7168"
}
```

**DeepSeek V3 支持**：

虽然架构枚举中没有 `LLM_ARCH_DEEPSEEK3`，但 llama.cpp 通过以下方式支持 DeepSeek V3：

1. **Chat Template**：
   - `--chat-template deepseek3` - 支持 DeepSeek V3 聊天模板
   - 模板文件：`models/templates/deepseek-ai-DeepSeek-V3.1.jinja`

2. **Chat Format**：
   - `COMMON_CHAT_FORMAT_DEEPSEEK_V3_1` - V3.1 格式支持
   - 支持 reasoning content 解析（`<think>` 标签）

3. **Tokenizer**：
   - `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM` - DeepSeek V3 tokenizer
   - 自动识别：`tokenizer_pre == "deepseek-v3"`

4. **模型转换**：
   - `convert_hf_to_gguf.py` 支持 `DeepseekV3ForCausalLM`
   - 自动识别为 `deepseek-v3` 架构

5. **MoE 支持**：
   - DeepSeek V3 的 MoE 架构（专家选择偏置）
   - 代码位置：`src/llama-graph.cpp:1150` - "add experts selection bias - introduced in DeepSeek V3"

**结论**：DeepSeek V3 在 llama.cpp 中通过 DeepSeek V2 的架构实现（`LLM_ARCH_DEEPSEEK2`），但使用 V3 特定的 chat template 和 tokenizer。

## 使用示例

### 加载 DeepSeek V2 模型

```bash
# 使用 llama.cpp CLI
./llama-cli -m deepseek-v2-236b.gguf \
    --chat-template deepseek2 \
    -p "Hello, how are you?"
```

### 量化 DeepSeek 模型

```bash
# 转换为 Q4_0 量化
./llama-quantize deepseek-v2-236b.gguf \
    deepseek-v2-236b-q4_0.gguf Q4_0
```

## 总结

✅ **llama.cpp 完全支持 DeepSeek 模型**

1. **架构支持**：
   - ✅ DeepSeek（原始版本）- `LLM_ARCH_DEEPSEEK`
   - ✅ DeepSeek V2（MoE 架构）- `LLM_ARCH_DEEPSEEK2`
   - ✅ DeepSeek V3（通过 V2 架构 + V3 模板）- `deepseek-v3`

2. **功能支持**：
   - ✅ 标准 Transformer 层
   - ✅ MoE 专家路由
   - ✅ MLA 注意力
   - ✅ YaRN RoPE
   - ✅ 所有量化格式

3. **工具支持**：
   - ✅ Tokenizer
   - ✅ Chat Template
   - ✅ Reasoning Format

4. **与你的定义文件**：
   - ✅ DeepSeek V2 定义完全对应（`LLM_ARCH_DEEPSEEK2`）
   - ✅ DeepSeek V3 定义对应（使用 `LLM_ARCH_DEEPSEEK2` 架构，V3 特定模板）

## 建议

1. **验证 DeepSeek V3 支持**：
   ```bash
   grep -r "DEEPSEEK3\|deepseek3" llama.cpp/src/
   ```

2. **测试你的定义文件**：
   - 使用 llama.cpp 加载 DeepSeek V2 模型
   - 验证 GEMM 操作与定义文件一致

3. **MoE 特定优化**：
   - DeepSeek V2 的 MoE 架构可能需要特殊的 GEMM 优化
   - 考虑专家并行计算
