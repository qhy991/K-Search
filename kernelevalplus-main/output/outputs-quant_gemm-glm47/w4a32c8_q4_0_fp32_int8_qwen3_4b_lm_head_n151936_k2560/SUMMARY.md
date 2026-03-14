# W4A32C8 Q4_0 Quantized GEMM - Final Summary

**Task**: `w4a32c8_q4_0_fp32_int8_qwen3_4b_lm_head_n151936_k2560`
**Target**: RTX 4090 (CC 8.9, 128 SMs)

## Problem Dimensions
- N = 151936 (output features, vocab size)
- K = 2560 (input features, hidden size)
- M = variable (batch size: 1, 2, 3, 4, 5, 8, 512)

## Quantization Format
- **Weights**: Q4_0 (4-bit, block size 32)
  - Block structure: 18 bytes (2-byte FP16 scale + 16 bytes packed 4-bit values)
  - Encoding: q = round(val / scale + 8), q in [0, 15]
  - Decoding: val = scale * (q - 8), val in [-8*scale, 7*scale]

- **Activation**: FP32 (not quantized)

## Computation Formula
```
output[m,n] = sum_b(sum_i(d_w[n,b] * (q_w[n,b,i] - 8) * act[m,k]))
```

## Performance Results

### Final Combined Kernel (Strategy Dispatch)

| M (batch) | Latency (ms) | TFLOPS | Notes |
|------------|---------------|--------|-------|
| 1          | 0.784         | 0.992  | Best single-token |
| 8          | 1.707         | 3.646  | Small batch kernel |
| 512        | 134.714        | 2.957  | Large batch kernel |

### Baseline Comparison
- Baseline (GGML/RTX4090): ~224.44 TFLOPS for similar configuration
- Our kernel achieves ~1.3% of baseline
- **Gap**: The W4A32 format with FP32 activation requires `d_w * (q - 8) * act` computation
  - This is essentially FP32 multiply for each element pair
  - Without tensor cores or integer dot products (INT8), we cannot reach baseline performance

## Key Optimizations
1. **Strategy Dispatch**: Different kernels for small vs large M
   - Small M (<=8): Simple kernel, 16x16 threads, better for single-token
   - Large M (>8): 2-way unrolled kernel, 32x8 threads, better for throughput

2. **Q4_0 Decoding**: Applied offset compensation within loop
   - `w_low = (byte_val & 0x0F) - 8`
   - `w_high = ((byte_val >> 4) & 0x0F) - 8`

3. **2-way K-block unrolling**: Process 2 weight blocks per outer iteration
   - Better instruction-level parallelism (ILP)
   - Reduced loop overhead

## Roofline Analysis

**Operational Intensity (OI)**:
- M=1: OI ≈ 12.7 FLOPs/Byte (MEMORY-BOUND)
- M=512: OI ≈ 1057 FLOPs/Byte (COMPUTE-BOUND)

**RTX 4090 Ridge Point (FP32)**:
- Ridge Point ≈ 81.9 FLOPs/Byte
- M=1: OI < Ridge → Memory-bound strategy
- M=512: OI > Ridge → Compute-bound strategy

**Implication**:
- Small M: Focus on bandwidth utilization, minimize thread count
- Large M: Focus on compute throughput, maximize occupancy
