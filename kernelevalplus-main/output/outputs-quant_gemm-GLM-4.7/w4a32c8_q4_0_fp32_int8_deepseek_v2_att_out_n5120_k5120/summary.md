# Q4_0 GEMM Kernel Optimization Summary
## DeepSeek-V2 Attention Output: w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120

### Task Overview
Implement and optimize a quantized GEMM kernel for DeepSeek-V2 attention output projection with Q4_0 weight quantization.

**Dimensions**: M (variable), N=5120, K=5120
**Quantization**: Q4_0 weights (4-bit, block size 32)
**Activation**: FP32 (dynamically quantized to Q8_1 during compute)

---

## Hardware Profile: RTX 4090
| Specification | Value |
|--------------|-------|
| Compute Capability | 8.9 |
| SM Count | 128 |
| Peak FP32 | 82.6 TFLOPS |
| Peak Bandwidth | 1008 GB/s |
| Shared Memory/Block | 48 KB |

---

## Roofline Analysis

**Operational Intensity (OI) = FLOPs / Bytes**

| Configuration | OI | Ridge Point | Bottleneck | Optimization Focus |
|--------------|-----|-------------|------------|-------------------|
| M=1 | 3.2 FLOPs/Byte | 0.1 | Compute | ILP, loop unrolling |
| M=8 | 25.1 FLOPs/Byte | 0.1 | Compute | Vectorized ops |
| M=512 | 718.6 FLOPs/Byte | 0.1 | Compute | Throughput |

**All configurations are compute-bound** → Focus on arithmetic throughput and ILP.

---

## Optimization History

### Version 1-2: Initial Implementation (FAILED)
- **Approach**: Basic Q4_1-style kernels adapted for Q4_0
- **Issues**: Incorrect dispatch logic, grid dimension errors
- **Result**: ❌ Correctness failure for M>1

### Version 3: First Working Version ✅
- **Approach**: Simple thread-per-output kernels
- **Kernels**:
  - `q4_0_gemm_single_token`: Thread-per-output for M=1
  - `q4_0_gemm_simple`: Thread-per-output for M≥8
  - `q4_0_gemm_warp`: Warp-based for 2≤M<8
- **Results**:
  - M=1: 0.201 ms, 0.26 TFLOPS
  - M=8: 0.401 ms, 1.047 TFLOPS
  - M=512: 21.551 ms, 1.246 TFLOPS
- **Status**: ✅ All correct (NMSE=0)

### Version 4-5: ILP Optimization (MIXED)
- **Approach**: 4-way block processing for ILP
- **Issues**:
  - Shared memory bank conflicts in warp kernel
  - Excessive register pressure
- **Results**: Worse for large M (0.5 TFLOPS)

### Version 6: Warp Kernel Optimization ✅
- **Approach**: Optimized warp kernel with 2-block processing
- **Results**: Similar to v3 (no significant improvement)

### Version 7: Multi-Output Processing (MIXED)
- **Approach**: 4 outputs per thread for M=1, 8 per block for large M
- **Results**:
  - M=1: 1.283 ms, 0.04 TFLOPS ❌ (worse)
  - M=8: 0.328 ms, 1.277 TFLOPS ✅ (+22%)
  - M=512: 17.908 ms, 1.499 TFLOPS ✅ (+20%)

### Version 8: Hybrid Strategy (BEST) ✅✅
- **Approach**: Best kernels from each version combined
  - M=1: Simple thread-per-output (from v3)
  - 2≤M<8: 2 outputs per thread (from v7)
  - M≥8: 8 outputs per block (from v7)
- **Results**:
  - M=1: 0.202 ms, 0.26 TFLOPS ✅
  - M=8: 0.328 ms, 1.28 TFLOPS ✅ (+22% vs v3)
  - M=512: 17.886 ms, 1.50 TFLOPS ✅ (+20% vs v3)

---

## Performance Comparison

| Version | M=1 (TFLOPS) | M=8 (TFLOPS) | M=512 (TFLOPS) | Status |
|---------|--------------|--------------|----------------|--------|
| v3 | 0.26 | 1.05 | 1.25 | ✅ Baseline |
| v6 | 0.26 | 1.05 | 1.25 | ✅ Same |
| v7 | 0.04 | 1.28 | 1.50 | ⚠️ M=1 failed |
| **v8** | **0.26** | **1.28** | **1.50** | ✅✅ Best |

**Improvement**: +22% (M=8), +20% (M=512) vs baseline

---

## Q4_0 Format Specification

### Block Structure
```c
typedef struct {
    half d;         // FP16 scale (delta), 2 bytes
    uint8_t qs[16]; // 32 x 4-bit values packed, 16 bytes
} block_q4_0;       // Total: 18 bytes
```

### Encoding/Decoding
- **Encoding**: `q = round(val / scale + 8)`, q ∈ [0, 15]
- **Decoding**: `val = scale * (q - 8)`

### Packing Format (llama.cpp)
```
Positions 0-15:  Low nibbles of bytes 0-15
Positions 16-31: High nibbles of bytes 0-15
```

### Kernel Formula
```
output[m, n] = Σ(scale_w[n,kb] * (q_w[n,kb,i] - 8) * activation[m,kb*32+i])
```

---

## Best Implementation: v8

### Kernel Strategy
```c
if (M == 1) {
    // Simple thread-per-output: 256 threads/block
    q4_0_gemm_single_token<<<blocks, 256>>>(...);
} else if (M >= 8) {
    // 8 outputs per block: 8 threads/block
    q4_0_gemm_large_batch_optimized<<<grid, 8>>>(...);
} else {
    // 2 outputs per thread: 128 threads/block
    q4_0_gemm_medium_batch<<<grid, 128>>>(...);
}
```

### Key Optimizations
1. **ILP**: `#pragma unroll 4` on inner loops
2. **Multi-output**: 2-8 outputs per thread/block
3. **Adaptive configuration**: Different kernels for different M
4. **Efficient dequantization**: Inline FP16→FP32 conversion

---

## Correctness Verification

All test configurations pass with **NMSE = 0.0** (threshold: 0.05):

```
single_token: ✅ NMSE=0.000000
small_batch: ✅ NMSE=0.000000
large_batch: ✅ NMSE=0.000000
```

---

## Performance Metrics

### Efficiency vs Peak
| Config | Achieved | Peak | % Peak |
|--------|----------|------|--------|
| M=1 | 0.26 TFLOPS | 82.6 TFLOPS | 0.31% |
| M=8 | 1.28 TFLOPS | 82.6 TFLOPS | 1.55% |
| M=512 | 1.50 TFLOPS | 82.6 TFLOPS | 1.82% |

**Note**: Low efficiency is typical for quantized GEMM due to:
- Dequantization overhead (FP16 read + unpack + subtract + multiply)
- Irregular memory access patterns
- Limited use of tensor cores (requires INT8)

---

## Conclusion

The v8 hybrid kernel achieves **20%+ improvement** over baseline for small and large batches while maintaining correctness. The key insight is that different batch sizes require different optimization strategies.
