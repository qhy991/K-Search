# W4A32C8 Q4_0 Quantized GEMM - DeepSeek-V2 MoE Down Projection

## Overview

This CUDA kernel implements an optimized quantized GEMM operation for the DeepSeek-V2 MoE shared expert down projection layer. It uses the BLOCK_Q4_0 × Q8_1 pattern compatible with llama.cpp.

## Specifications

| Parameter | Value |
|-----------|-------|
| N (Output features) | 5120 |
| K (Input features) | 12288 |
| M (Batch dimension) | Variable (1-512 tested) |
| Block size | 32 |
| Weight format | BLOCK_Q4_0 (4-bit quantization) |
| Activation format | FP32 |

## Performance

| Config | Latency | TFLOPS | GFLOPS | vs Peak |
|--------|---------|--------|--------|---------|
| M=1 | 1.72 ms | 0.073 | 73.1 | 0.09% |
| M=8 | 0.92 ms | 1.092 | 1091.8 | 1.32% |
| M=512 | 34.72 ms | 1.856 | 1855.6 | 2.25% |

## Compilation

```bash
cd /home/qinhaiyan/kernelevalplus

# Using setup.py
python setup.py build_ext --inplace

# Or using JIT compilation (automatic in test framework)
python llm_kernel_test/unified_test_runner.py --test \
    --definition definitions/quant_gemm/deepseek_v2/w4a32c8_q4_0_fp32_int8_ds2_moe_down_n5120_k12288.json \
    --attempt-path output/outputs-quant_gemm-GLM-4.7/w4a32c8_q4_0_fp32_int8_ds2_moe_down_n5120_k12288
```

## Testing

```bash
# Run correctness and performance tests
python llm_kernel_test/unified_test_runner.py --test \
    --definition definitions/quant_gemm/deepseek_v2/w4a32c8_q4_0_fp32_int8_ds2_moe_down_n5120_k12288.json \
    --attempt-path attempts/w4a32c8_q4_0_fp32_int8_ds2_moe_down_n5120_k12288_v12
```

## Usage

```python
import torch
from pathlib import Path

# Load the compiled kernel
kernel_lib = torch.utils.cpp_extension.load(
    name="w4a32c8_q4_0_fp32_int8_ds2_moe_down_n5120_k12288",
    sources=["kernel_best.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    extra_cflags=["-O3"],
)

# Prepare inputs
M, N, K = 1, 5120, 12288  # Adjust M as needed
weight_q4 = torch.randint(0, 256, (N, K//32 * 18), dtype=torch.uint8, device="cuda")
activation = torch.randn(M, K, dtype=torch.float32, device="cuda")

# Call the kernel
output = kernel_lib.forward(weight_q4, activation, M, N, K)

print(f"Output shape: {output.shape}")  # [M, N]
```

## Implementation Details

### Q4_0 Format

- **Size**: 34 bytes per 32 values
- **Scale**: 2 bytes (FP16)
- **Data**: 16 bytes containing 32 packed 4-bit values
- Each 4-bit value represents -8 to +7 (stored as 0-15)

### Memory Layout

```
Weight: [N, K/32, 18]
- Each row: (K/32) blocks × 18 bytes = 6912 bytes (for K=12288)
- Block layout: [scale(2), packed_qs(16)]
```

### Kernel Strategy

The kernel uses different strategies based on batch size M:

1. **M=1**: 1024 threads per row, maximize parallelism
2. **M=2-8**: 512 threads per row, balanced configuration
3. **M=9+**: 128×8 2D tiling, optimized for compute-bound

### Key Optimizations

1. **Vectorized unpacking**: Process 8 packed 4-bit values per iteration
2. **Strategy dispatch**: Different implementations for different M
3. **Register optimization**: const vs non-const for different workloads
4. **Loop unrolling**: #pragma unroll for better instruction scheduling

## Files

- `kernel_best.cu` - Optimized CUDA kernel implementation
- `test_results.json` - Test results and performance metrics
- `summary.md` - Detailed optimization summary
- `performance_comparison.md` - Version-by-version comparison

## References

- [llama.cpp quantization](https://github.com/ggerganov/llama.cpp)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [DeepSeek-V2 Paper](https://github.com/deepseek-ai/DeepSeek-V2)
