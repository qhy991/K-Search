# llama.cpp 操作定义

本目录包含与 llama.cpp 兼容的 CUDA 内核的 JSON 模式定义，遵循 flashinfer-bench 格式。

## 目录结构

```
definitions/
├── quant_gemm/           # 量化 GEMM 操作
│   ├── w4a8_q4_0_q8_1_n4096_k4096.json
│   ├── w4a16_q4_0_fp32_n4096_k4096.json
│   └── w4_1a8_q4_1_q8_1_n4096_k4096.json
├── quant_vec_dot/        # 分块点积
│   └── vec_dot_q4_0_q8_1.json
├── quantize/             # 量化操作
│   └── quantize_q8_1_k4096.json
├── rmsnorm/              # RMS 归一化
├── activation/           # 激活函数
└── README.md
```

## JSON 模式格式

每个定义遵循以下结构：

```json
{
  "name": "operation_name",
  "op_type": "quant_gemm | quantize | quant_vec_dot | ...",
  "variant": "W4A8 | q8_1 | ...",
  "description": "Human-readable description",
  "tags": ["status:verified", "framework:llama.cpp", "quantization:q4_0"],

  "axes": {
    "M": {"type": "var", "description": "..."},
    "N": {"type": "const", "value": 4096, "description": "..."}
  },

  "inputs": {
    "tensor_name": {
      "shape": ["M", "K"],
      "dtype": "float32 | block_q4_0 | block_q8_1",
      "description": "..."
    }
  },

  "outputs": {
    "tensor_name": {
      "shape": ["M", "N"],
      "dtype": "float32",
      "description": "..."
    }
  },

  "constraints": ["K % 32 == 0", "sizeof(block_q4_0) == 18"],

  "types": {
    "block_q4_0": {
      "size": 18,
      "fields": [...]
    }
  },

  "formula": {
    "dot_product": "result = d_w * (d_a * sumi - 8.0f * s_a)",
    "explanation": "..."
  },

  "reference": "Python reference implementation as string"
}
```

## 与 flashinfer-bench 的主要区别

| 特性 | flashinfer-bench | llama.cpp (本实现) |
|---------|-----------------|------------------|
| 数据类型 | float16, float8_e4m3fn | block_q4_0, block_q8_1 等 |
| 块结构 | 无 | 显式类型定义 |
| 公式 | 简单 GEMM | 补偿公式 |
| 量化 | 块级 FP8 | 每块对称/非对称 |

## 支持的量化类型

| 类型 | 大小 | 描述 | 主要用途 |
|------|------|-------------|-------------|
| block_q4_0 | 18 字节 | 4 位对称，偏移 +8 | 权重 |
| block_q4_1 | 20 字节 | 4 位非对称，最小-最大 | 权重 |
| block_q5_0 | 22 字节 | 5 位对称 | 权重 |
| block_q5_1 | 24 字节 | 5 位非对称 | 权重 |
| block_q8_0 | 34 字节 | 8 位对称 | 权重 |
| block_q8_1 | 36 字节 | 8 位带求和 | 激活 |

## 命名约定

```
{op}_{variant}_{weight_type}_{act_type}_n{N}_k{K}.json

示例:
- w4a8_q4_0_q8_1_n4096_k4096.json   # W4A8，N=4096，K=4096
- w4a16_q4_0_fp32_n4096_k4096.json  # W4A16，FP32 激活
- quantize_q8_1_k4096.json           # Q8_1 量化
```

## 用于 LLM 代码生成的用法

1. **选择定义**：选择与目标操作匹配的定义
2. **解析 JSON**：提取以下信息：
   - 输入/输出张量形状和类型
   - 类型定义（块结构）
   - 计算公式
   - 参考实现
3. **生成 CUDA 内核**：按照规范生成

示例提示：
```
使用 w4a8_q4_0_q8_1_n4096_k4096.json 中的定义：
1. 生成一个简单的 CUDA 内核
2. 使用精确公式：d_w * (d_a * sumi - 8.0f * s_a)
3. 确保 block_q4_0 为 18 字节，block_q8_1 为 36 字节
4. 为 M、N、K 添加适当的边界检查
```

## 验证

每个定义包括：
- `constraints` 字段中的 `static_assert` 大小约束
- `reference` 字段中的 Python 参考实现
- 预期相对误差阈值（通常 < 1e-3）
