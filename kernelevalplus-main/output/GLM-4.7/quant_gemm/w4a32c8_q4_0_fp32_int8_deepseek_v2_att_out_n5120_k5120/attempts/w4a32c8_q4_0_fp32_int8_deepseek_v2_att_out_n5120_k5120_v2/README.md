# W4A32C8 Q4_0 Quantized GEMM for DeepSeek V2 Attention Output

## Summary

Successfully implemented a Q4_0 quantized GEMM kernel for DeepSeek V2 attention output projection (N=5120, K=5120, variable M). The kernel uses llama.cpp-compatible Q4_0 format with FP32 activations and implements strategy dispatch for optimal performance across different batch sizes.

## Performance Results

| Configuration | M | Latency (ms) | TFLOPS | Status |
|--------------|---|--------------|--------|--------|
| single_token  | 1 | 0.260        | 0.20   | ✅ Pass |
| small_batch   | 8 | 2.008        | 0.21   | ✅ Pass |
| large_batch   | 512 | 11.515      | 2.33   | ✅ Pass |

**Correctness**: All tests pass with NMSE ≈ 0.0

## Implementation Details

### Q4_0 Format (llama.cpp compatible)
- **Block size**: 32 values per block
- **Storage**: 18 bytes per block (2 bytes FP16 scale + 16 bytes packed quaternions)
- **Encoding**: `q = round(val / scale + 8)`, where q ∈ [0, 15]
- **Decoding**: `val = scale × (q - 8)`
- **Unpacking**:
  - Positions 0-15: `packed[i] & 0x0F`
  - Positions 16-31: `(packed[i] >> 4) & 0x0F`

### Strategy Dispatch Pattern

The implementation uses different kernels based on the batch size (M):

1. **Small M (≤8)**: Simple kernel with one thread per output element
   - Grid: `(N, M)` blocks
   - Block: `1` thread
   - Minimal overhead for small batches

2. **Large M (>8)**: Optimized kernel with 8×8 thread blocks
   - Grid: `((N+7)/8, (M+7)/8)` blocks
   - Block: `(8, 8)` threads
   - Better parallelism and memory utilization for large batches

### Key Optimizations

1. **Vectorized unpacking**: Process both nibbles in single loop iteration
2. **Union-based FP16 conversion**: Safe cross-platform FP16 to FP32 conversion
3. **Coalesced memory access**: Sequential weight block access patterns
4. **Loop unrolling**: `#pragma unroll` for inner computation loops

## Bug Fixes

### Handler Dtype Format Issue

**Problem**: The `QuantGEMMHandler` in `llm_kernel_test/op_test_handler.py` was checking for `"block_q4_0"` dtype but the spec uses `"q4_0"`, causing it to fall back to Q8_0 format (34 bytes/block instead of 18 bytes/block).

**Fix**: Updated handler to support both formats:
```python
if weight_dtype in ("q4_0", "block_q4_0"):
    weight_q_bytes = quantize_to_q4_0(weight_fp32)
```

This fix was applied in three locations:
- `generate_inputs()` method
- `get_reference_output()` method
- Activation dtype checking for Q8_1

## Files

### Kernel Implementation
- `attempts/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_v2/kernel.cu`
  - Final optimized version with strategy dispatch
  - Passes all tests with NMSE ≈ 0

### Test Framework
- `llm_kernel_test/op_test_handler.py` (modified)
  - Fixed dtype format checking for Q4_0/Q8_0/Q8_1

### Definition
- `definitions/quant_gemm/deepseek_v2/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120.json`
  - Operator specification
  - Test configurations: M ∈ {1, 2, 3, 4, 5, 8, 512}

## Roofline Analysis (RTX 4090)

- **Peak FP32 TFLOPS**: 82.6
- **Peak Memory Bandwidth**: 1008 GB/s
- **Ridge Point**: 0.082 FLOPs/Byte

Due to Q4_0 compression (15.6% of FP32 size), all cases are compute-bound with OI ranging from 3.19 (M=1) to 718.6 (M=512).

## Usage

```python
import torch
from llm_kernel_test.unified_test_runner import UnifiedTestRunner

runner = UnifiedTestRunner()
results = runner.test(
    attempt_id="w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_v2",
    definition_path="definitions/quant_gemm/deepseek_v2/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120.json",
    attempt_path="attempts/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_v2"
)
```

## Future Optimization Opportunities

1. **Tensor Core utilization**: Use MMA/WMMA instructions for INT8 matrix multiply-accumulate
2. **Shared memory tiling**: Cache activation blocks in shared memory
3. **Pipeline stages**: Overlap memory loads with computation
4. **Warp-level aggregation**: Use warp shuffle for partial sum reduction

## References

- llama.cpp Q4_0 format: https://github.com/ggerganov/llama.cpp
- Roofline model: Williams, Waterman, Patterson (2009)
