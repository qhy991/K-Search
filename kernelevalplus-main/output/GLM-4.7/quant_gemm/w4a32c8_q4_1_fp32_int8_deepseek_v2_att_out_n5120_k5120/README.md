# W4A32C8 Q4_1 FP32 INT8 DeepSeek-V2 Attention Output

## 最佳版本

**Version**: v8
**File**: `kernel.cu`
**Performance**:
- M=1: 0.279 TFLOPS
- M=8: 0.647 TFLOPS
- M=512: 1.20 TFLOPS

## 快速开始

### 编译和测试
```bash
# 运行测试
python llm_kernel_test/unified_test_runner.py --test \
    --definition definitions/quant_gemm/deepseek_v2/w4a32c8_q4_1_fp32_int8_deepseek_v2_att_out_n5120_k5120.json \
    --attempt-path output/GLM-4.7/quant_gemm/w4a32c8_q4_1_fp32_int8_deepseek_v2_att_out_n5120_k5120
```

### 使用内核
```python
import torch
import torch.utils.cpp_extension

# 编译
torch.utils.cpp_extension.load(
    name='w4a32c8_q4_1_fp32_int8_deepseek_v2_att_out_n5120_k5120',
    sources=['kernel.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_cxx_cflags=['-O3'],
    is_python_module=True,
    verbose=True
)

# 使用
from w4a32c8_q4_1_fp32_int8_deepseek_v2_att_out_n5120_k5120 import forward

output = forward(weight, activation, M, N, K)
```

## 文件说明

- `kernel.cu` - CUDA内核实现 (最佳版本 v8)
- `test_results.json` - 性能测试结果
- `summary.md` - 详细优化总结文档
- `README.md` - 本文件

## 技术规格

- **量化格式**: Q4_1 (4-bit非对称量化)
- **激活格式**: FP32 (动态量化为Q8_1)
- **块大小**: 32元素
- **目标硬件**: NVIDIA RTX 4090 (Compute 8.9)

## 性能基线

| M | 本实现 | GGML基线 | 差距 |
|---|--------|---------|------|
| 1 | 0.28 TFLOPS | 6.98 TFLOPS | 25x |
| 512 | 1.20 TFLOPS | 199.43 TFLOPS | 166x |
