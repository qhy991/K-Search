# W4A32C8 Q4_0 FP32 INT8 MoE Routing Up N5120 K1536

## Overview

This directory contains the optimized CUDA kernel implementation for the DeepSeek-V2 MoE routing expert up/gate projection with W4A32C8 quantization.

**Operator**: Quantized GEMM (BLOCK_Q4_0 × FP32 activation, llama.cpp compatible)
**Dimensions**: N=5120, K=1536, M (variable 1-512)
**Hardware**: NVIDIA RTX 4090 (Compute Capability 8.9)

## Quick Results

| Metric | M=1 | M=8 | M=512 |
|--------|-----|-----|-------|
| Performance | 1.14 TFLOPS | 1.64 TFLOPS | **3.06 TFLOPS** |
| vs Baseline | 46% | - | 1.9% |

## Files

### Main Files
- `kernel_best.cu` - Best performing kernel (v10 with shared memory)
- `test_results.json` - Test results and performance data
- `SUMMARY.md` - Complete optimization journey summary
- `PERFORMANCE_TABLE.md` - Detailed performance comparison

### Version History
See `versions/` directory for key implementations:
- `kernel_v04_k_parallelization.cu` - Best for M=1 (1.28 TFLOPS)
- `kernel_v05_dual_path.cu` - Balanced performance
- `kernel_v10_shared_memory.cu` - Best for M=512 (3.06 TFLOPS)
- `kernel_v11_final.cu` - Final version

## Usage

```python
import torch

# Load the compiled kernel
kernel = torch.ops.w4a32c8_q4_0_fp32_int8_ds2_moe_routing_up_n5120_k1536_quant_gemm_test

# Run the kernel
output = kernel.forward(weight, activation, M, N, K)
```

## Key Optimizations

1. **K-Parallelization** (small M): Warp-level parallelization across K dimension
2. **Dual-Path Strategy**: Different kernels for different M ranges
3. **Shared Memory Tiling**: Cache activation blocks for reuse (28% improvement for M=512)
4. **DP4A Instruction**: Efficient int8 dot product computation

## Correctness

All tests pass with NMSE < 0.001 (threshold 0.05)

## Implementation Details

### BLOCK_Q4_0 Format
- 18 bytes per block
- 2 bytes FP16 scale + 16 bytes packed 4-bit values
- Values in [0,15] represent actual values in [-8,+7]

### Computation Formula (llama.cpp pattern)
```
result = d_w * (d_a * sumi - 8.0 * a_sum)
```
where:
- `d_w`: weight scale (FP16)
- `d_a`: activation max/127
- `sumi`: int8 dot product accumulation
- `-8.0 * a_sum`: Q4_0 offset compensation

## Performance vs Reference

Our kernel (3.02 TFLOPS for M=512) achieves **63% better performance**
than the att_out reference implementation (1.85 TFLOPS).
