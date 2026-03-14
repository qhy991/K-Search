# W4A32C8 Q4_0 × FP32 Quantized GEMM - DeepSeek-V3 Attention Output

**Problem ID**: `w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168`

## Overview

This kernel implements a quantized General Matrix Multiply (GEMM) operation for DeepSeek-V3's attention output projection. The operation computes `C = A @ W^T` where:
- **A** (activation): FP32 tensor of shape [M, K]
- **W** (weight): Q4_0 quantized tensor of shape [N, K/32]
- **C** (output): FP32 tensor of shape [M, N]

## Dimensions

| Axis | Value | Description |
|------|-------|-------------|
| M | Variable (1-512) | Batch dimension |
| N | 7168 | Output features |
| K | 7168 | Input features |
| Block Size | 32 | Quantization block size |

## Quantization Format

### Q4_0 Weight Format (18 bytes per block)
- **d** (2 bytes): FP16 scale/dequantization factor
- **qs** (16 bytes): 32 packed 4-bit values

**Unpacking (llama.cpp compatible)**:
- Positions 0-15: All low nibbles from bytes 0-15
- Positions 16-31: All high nibbles from bytes 0-15

### Q8_1 Dynamic Quantization on Activation
Applied on-the-fly during computation:
1. Find max absolute value in each 32-element block
2. Scale: `d_a = max_abs / 127`
3. Sum: `s_a = sum(activation)`
4. Quantized: `q_a[i] = round(activation[i] / d_a)`

### Computation Formula
```
result = d_w * (d_a * sumi - 8 * s_a)
```
where `sumi = dot(q_a, q_w)` computed using DP4A instruction.

## Implementation Strategy

The kernel uses **adaptive strategy dispatch** based on batch size:

### Small Batch (M ≤ 8): Split-K Kernel
- Parallelizes K dimension across multiple thread blocks
- Each thread processes multiple blocks
- Warp-level reduction followed by atomic accumulation
- Dynamic split count based on SM count

### Large Batch (M > 8): Tiled Kernel
- Tile size: TILE_M=4, TILE_N=64
- 256 threads per block (8 warps)
- Each thread computes one output element

## Hardware Optimizations

1. **DP4A Intrinsic**: Hardware-accelerated 4-bit dot product (CC ≥ 6.1)
2. **__forceinline__**: Aggressive inlining for device functions
3. **Union-based FP16 conversion**: Efficient half to float conversion
4. **Vectorized memory access**: float4 loads for activation data
5. **Loop unrolling**: #pragma unroll for critical loops
6. **torch::empty**: Avoid zero-initialization overhead

## Performance Results

| Config | Latency | TFLOPS | vs Best | Correctness (NMSE) |
|--------|---------|--------|---------|-------------------|
| M=1    | 0.056 ms | 1.821 | 0.2% slower | 3.8e-05 ✅ |
| M=8    | 0.416 ms | 1.977 | 0.1% slower | 0.000514 ✅ |
| M=512  | 23.57 ms | 2.232 | 4% slower | 0.000233 ✅ |

## Performance Comparison

| Version | M=1 (TFLOPS) | M=8 (TFLOPS) | M=512 (TFLOPS) |
|---------|--------------|--------------|----------------|
| v1 (basic) | 0.754 | 1.792 | 2.270 |
| v2 (split-K) | 1.826 | 1.991 | 2.221 |
| **v3 (final)** | **1.821** | **1.977** | **2.232** |
| Best-known | 1.825 | 1.979 | 2.320 |

## Key Improvements Through Iterations

### v1 → v2 (+140% for M=1)
- Added Split-K strategy for small batches
- Implemented warp-level reduction
- Fixed Q4_0 unpacking format (llama.cpp compatible)

### v2 → v3 (numerical stability)
- Fixed `inv_d_a` calculation: `1.0f / d_a` instead of `127.0f / a_max`
- Added `__forceinline__` to device functions
- Used union for FP16 conversion
- Switched to `torch::empty` for output

## Files

- **kernel.cu**: Main implementation
- **test_results.json**: Test results and benchmarks

## Usage

```python
import torch

# Load the compiled extension
import w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_quant_gemm_test

# Forward pass
output = w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_quant_gemm_test.forward(
    weight,    # [N, K/32*18] uint8 tensor (Q4_0 blocks)
    activation, # [M, K] float32 tensor
    M, N, K     # dimensions
)
```

## Correctness Verification

All test cases pass with NMSE well below the 0.05 threshold:
- ✅ single_token (M=1)
- ✅ small_batch (M=8)
- ✅ large_batch (M=512)

## Hardware Requirements

- Compute Capability ≥ 6.1 (for DP4A instruction)
- Tested on NVIDIA RTX 4090 (CC 8.9)
- 128 SMs, 23.6 GB memory

## References

- llama.cpp Q4_0 quantization format
- CUDA DP4A intrinsic documentation
- GGML baseline implementation
