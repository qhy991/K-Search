# W4A32C8 Q4_0 Quantized GEMM Optimization Summary

## Task Overview

**Target**: DeepSeek V2 Attention Output Projection
- Quantization: W4A32C8 (Q4_0 weights, FP32 activations)
- Dimensions: N=5120, K=5120, variable M (1-512)
- Format: llama.cpp compatible Q4_0

## Hardware Specifications (RTX 4090)

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4090 |
| Compute Capability | 8.9 |
| SM Count | 128 |
| Global Memory | 23.6 GB |
| Peak FP32 TFLOPS | 82.6 |
| Peak Memory Bandwidth | 1008 GB/s |
| Ridge Point | 0.082 FLOPs/Byte |

## Roofline Analysis

### Operational Intensity (OI = FLOPs/Bytes)

For Q4_0 format (20 bytes per 32 values vs 128 bytes for FP32):

| M | OI (FLOPs/Byte) | Binding | Predicted TFLOPS |
|---|------------------|---------|------------------|
| 1 | 3.19 | Compute-bound | 82.6 |
| 8 | 25.10 | Compute-bound | 82.6 |
| 512 | 718.60 | Compute-bound | 82.6 |

**Key Insight**: All cases are compute-bound due to Q4_0 compression. Focus should be on maximizing arithmetic throughput.

## Optimization Journey

### Version 1: Initial Simple Implementation
**Status**: ❌ Failed (NaN output)

**Approach**:
- One thread per output element
- Basic Q4_0 unpacking
- Direct FP16 to FP32 conversion

**Issue**: NaN output due to incorrect FP16 conversion (`__half` struct doesn't have `.x` member in CUDA 12.8)

### Version 2: Corrected Simple Implementation
**Status**: ✅ Pass

**Performance**:
| M | Latency (ms) | TFLOPS |
|---|--------------|--------|
| 1 | 0.260 | 0.20 |
| 8 | 2.008 | 0.21 |
| 512 | 116.254 | 0.23 |

**Key Fixes**:
1. Fixed FP16 conversion using union: `union { uint16_t u16; __half f16; }`
2. Correct Q4_0 unpacking: low nibbles (0-15), high nibbles (16-31)
3. Fixed test handler dtype format bug (`"q4_0"` vs `"block_q4_0"`)

**Limitation**: Poor performance for large M due to insufficient parallelism.

### Version 3: Strategy Dispatch Implementation
**Status**: ✅ Pass

**Performance**:
| M | Latency (ms) | TFLOPS | Speedup |
|---|--------------|--------|---------|
| 1 | 0.262 | 0.20 | 1.0x |
| 8 | 2.014 | 0.21 | 1.0x |
| 512 | 11.525 | 2.33 | **10.1x** |

**Optimizations**:
1. **Strategy Dispatch**: Different kernels for different M values
   - Small M (≤8): 1 thread per element
   - Large M (>8): 8×8 thread blocks
2. **Vectorized Unpacking**: Process both nibbles per iteration
3. **Coalesced Access**: Sequential weight block access

## Best Version: Final Strategy Dispatch

**File**: `attempts/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_v2/kernel.cu`

### Key Features

1. **Dual Kernel Strategy**:
```cuda
if (M <= 8) {
    // Simple kernel: dim3 grid(N, M), dim3 block(1)
} else {
    // Optimized kernel: dim3 grid((N+7)/8, (M+7)/8), dim3 block(8, 8)
}
```

2. **Efficient Unpacking**:
```cuda
for (int i = 0; i < 16; i++) {
    uint8_t packed = wb.qs[i];
    int q0 = packed & 0x0F;           // Position i (0-15)
    int q1 = (packed >> 4) & 0x0F;    // Position i+16 (16-31)
    // Process both values
}
```

3. **Safe FP16 Conversion**:
```cuda
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}
```

## Performance Comparison

| Version | M=1 TFLOPS | M=8 TFLOPS | M=512 TFLOPS | Best Config Speedup |
|---------|------------|------------|--------------|---------------------|
| v1 | ❌ NaN | ❌ NaN | ❌ NaN | - |
| v2 | 0.20 | 0.21 | 0.23 | 1.0x (baseline) |
| v3 (final) | 0.20 | 0.21 | **2.33** | **10.1x** for M=512 |

## Bug Fixes

### Test Handler Dtype Format Issue

**Location**: `llm_kernel_test/op_test_handler.py`

**Problem**: Handler checked for `"block_q4_0"` but spec uses `"q4_0"`

**Fix**: Support both formats
```python
if weight_dtype in ("q4_0", "block_q4_0"):
    weight_q_bytes = quantize_to_q4_0(weight_fp32)
```

Applied to:
- `generate_inputs()` - line 190
- `get_reference_output()` - line 241
- Activation Q8_1 checking - line 212

## Correctness Results

All test configurations pass with NMSE ≈ 0:

| Config | M | N | K | NMSE | Status |
|--------|---|---|---|------|--------|
| single_token | 1 | 5120 | 5120 | 0.000000 | ✅ |
| small_batch | 8 | 5120 | 5120 | 0.000000 | ✅ |
| large_batch | 512 | 5120 | 5120 | 0.000000 | ✅ |

## Technical Details

### Q4_0 Format Specification

**Block Structure** (18 bytes):
- `[0:2]`: FP16 scale (delta)
- `[2:18]`: 16 bytes packed quaternions

**Packing Format**:
- `byte[i] = q[i] | (q[i+16] << 4)`
- Low nibble: position i (0-15)
- High nibble: position i+16 (16-31)

**Quantization**:
- Encode: `q = round(val / scale + 8)`
- Decode: `val = scale × (q - 8)`

### Memory Layout

```
Weight: [N, K/32] blocks × 18 bytes = 14.7 MB for N=K=5120
Activation: [M, K] × 4 bytes = 0.02 MB (M=1) to 10.5 MB (M=512)
Output: [M, N] × 4 bytes = 0.02 MB (M=1) to 10.5 MB (M=512)
```

## Future Optimization Opportunities

1. **Tensor Cores**: Use WMMA for INT8 dot product
2. **Shared Memory**: Cache activation blocks
3. **Pipelining**: Overlap loads with compute
4. **Vector Loads**: Use float4/int4 for coalesced access

## Files

### Best Kernel
- `output/GLM-4.7/quant_gemm/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120/kernel.cu`

### Test Framework Fix
- `llm_kernel_test/op_test_handler.py`

### Definition
- `definitions/quant_gemm/deepseek_v2/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120.json`

## Conclusion

The final implementation achieves:
- **Correctness**: NMSE ≈ 0 across all configurations
- **Performance**: 2.33 TFLOPS for M=512 (10x improvement over baseline)
- **Strategy**: Adaptive kernel selection based on batch size

The strategy dispatch pattern proves effective for handling varying batch sizes, with simple kernels for small batches avoiding overhead and optimized kernels for large batches maximizing parallelism.
