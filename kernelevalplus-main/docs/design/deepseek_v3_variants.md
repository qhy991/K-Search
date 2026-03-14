# DeepSeek-V3 Quantized GEMM Variants

## Overview

This document describes the quantized GEMM variants implemented for DeepSeek-V3 inference, including their characteristics, use cases, and llama.cpp compatibility.

## Variant Comparison

### W4A32C32 Q4_0 FP32

| Property | Value |
|----------|-------|
| **Weights** | Q4_0 (4-bit) |
| **Activations** | FP32 (32-bit) |
| **Compute** | FP32 (32-bit) |
| **Memory** | Lowest (4-bit weights) |
| **Speed** | Baseline |
| **Accuracy** | High (FP32 compute) |
| **Status** | ✅ Verified |

**Use Cases**: Memory-constrained scenarios where FP32 activations are acceptable.

---

### W8A32C8 Q8_0 INT8 (Legacy)

| Property | Value |
|----------|-------|
| **Weights** | Q8_0 (8-bit) |
| **Activations** | FP32 (32-bit input, dynamically quantized) |
| **Activation Quantization** | **Per-row** (each row has one scale) |
| **Compute** | INT8 (8-bit) |
| **Memory** | Medium (8-bit weights) |
| **Speed** | 2-4x faster than FP32 |
| **Accuracy** | Good |
| **llama.cpp Compatible** | ❌ No |
| **Status** | ⚠️ Legacy - Consider migrating to W8A8C8 |

**Key Difference**: Uses **per-row** activation quantization, which is simpler but less accurate than per-block quantization.

**Formula**:
```
result = sum_k(quant_row(A[m,:]) * W_qs[n,k]) * d_w * d_a_per_row
```

---

### W8A8C8 Q8_0×Q8_1 (NEW - Recommended) ✨

| Property | Value |
|----------|-------|
| **Weights** | Q8_0 (8-bit) |
| **Activations** | FP32 (32-bit input, dynamically quantized) |
| **Activation Quantization** | **Per-block** (32 values share one scale) |
| **Compute** | INT8 (8-bit) with DP4A |
| **Memory** | Medium (8-bit weights) |
| **Speed** | 2-4x faster than FP32 |
| **Accuracy** | Better than per-row quantization |
| **llama.cpp Compatible** | ✅ Yes |
| **Status** | ✅ Verified |

**Key Features**:
- Matches llama.cpp `vec_dot_q8_0_q8_1` pattern
- Uses DP4A instruction for efficient INT8 dot product
- Per-block activation quantization for better accuracy
- Formula: `result = d8_0 * d8_1 * sumi`

**llama.cpp Compatibility**:
```cpp
// llama.cpp vec_dot_q8_0_q8_1_impl
template <int vdr>
__device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < vdr; ++i) {
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }
    return d8_0 * d8_1 * ((T) sumi);
}
```

---

## DeepSeek-V3 Scenarios

All variants support the following DeepSeek-V3 inference scenarios:

| Scenario | N (Output) | K (Input) | Description |
|----------|------------|-----------|-------------|
| `att_out` | 7168 | 7168 | Attention output projection |
| `att_qkv` | 21504 | 7168 | Attention QKV projection (3×7168) |
| `moe_up` | 18432 | 7168 | MoE up projection (expert FFN) |
| `moe_down` | 7168 | 18432 | MoE down projection (expert FFN) |

## File Organization

```
definitions/quant_gemm/deepseek_v3/
├── w4a32c32_q4_0_fp32_ds3_*.json      # W4A32C32 definitions
├── w8a32c8_q8_0_int8_ds3_*.json        # W8A32C8 (legacy) definitions
└── w8a8c8_q8_0_q8_1_ds3_*.json        # W8A8C8 (new) definitions

operators/quant_gemm/variants/deepseek_v3/
├── w4a32c32_q4_0_fp32/                 # W4A32C32 implementation
├── w8a32c8_q8_0_int8/                 # W8A32C8 (legacy) implementation
└── w8a8c8_q8_0_q8_1/                 # W8A8C8 (new) implementation
    ├── spec.json
    ├── kernel.cu
    ├── reference.py
    ├── bindings.cpp
    └── build.py
```

## Recommendation

**For new DeepSeek-V3 implementations, use W8A8C8 Q8_0×Q8_1** because:

1. ✅ **llama.cpp Compatible**: Matches the industry-standard llama.cpp pattern
2. ✅ **Better Accuracy**: Per-block quantization is more accurate than per-row
3. ✅ **DP4A Optimization**: Uses hardware-accelerated INT8 dot product instruction
4. ✅ **Verified**: All tests pass with NMSE < 1e-04

## Migration Guide

If you're currently using W8A32C8 Q8_0 INT8 (legacy):

```python
# Old (not llama.cpp compatible)
from w8a32c8_q8_0_int8.reference import run as gemm_legacy
output = gemm_legacy(activation, weight_q8_0)

# New (llama.cpp compatible) - Recommended
from w8a8c8_q8_0_q8_1.reference import run as gemm_llama
output = gemm_llama(activation, weight_q8_0)
```

The key difference:
- **Old**: Per-row activation quantization
- **New**: Per-block (32 values) activation quantization

## Test Results

```
╔══════════════════════════════════════════════════╗
║          DeepSeek-V3 Quick Test Suite            ║
╚══════════════════════════════════════════════════╝

llama.cpp compatibility: ✓ PASS (0.94% relative error)
W8A8C8 Q8_0×Q8_1 variant:    ✓ PASS (NMSE < 1e-04)
```

## References

- llama.cpp: https://github.com/ggerganov/llama.cpp
- Quantization types: `include/quant_types.h`
- Test suite: `python/test_deepseek_v3_quick.py`
