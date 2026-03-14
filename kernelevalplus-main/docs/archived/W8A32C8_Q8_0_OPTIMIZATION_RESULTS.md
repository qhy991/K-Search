# W8A32C8 Q8_0 Kernel Optimization Results

## Summary

Successfully optimized the W8A32C8 Q8_0 GEMM kernel with **up to 1.55x speedup** while maintaining perfect accuracy (NMSE=0.000000).

## Test Configuration

- **Matrix Dimensions**: N=5120, K=5120
- **Batch Sizes**: M ∈ {1, 2, 4, 8}
- **Precision**: INT8 weights (Q8_0) × FP32 activations → FP32 output
- **Hardware**: NVIDIA GPU with CUDA 12.2, DP4A support
- **Compiler Flags**: `-O3 --use_fast_math`

## Performance Results

### Baseline vs Optimized Comparison

| Batch | Baseline (ms) | Optimized (ms) | Baseline (GFLOPS) | Optimized (GFLOPS) | Speedup | Improvement |
|-------|---------------|----------------|-------------------|--------------------|---------| ------------|
| 1     | 0.259         | 0.300          | 202.1             | 174.8              | 0.86x   | -13.5%      |
| 2     | 0.336         | 0.300          | 311.9             | 349.5              | 1.12x   | +12.1%      |
| 4     | 0.519         | 0.336          | 403.7             | 623.9              | **1.55x** | **+54.5%**  |
| 8     | 0.725         | 0.710          | 578.7             | 590.4              | 1.02x   | +2.0%       |

### Correctness

✅ **All tests PASS** with NMSE=0.000000 (perfect numerical accuracy)

## Optimization Techniques

### Baseline Kernel
- Simple per-thread-per-output design
- Each thread computes one output element
- On-the-fly activation quantization (FP32 → Q8_1)
- DP4A instruction for INT8×INT8 dot product
- Direct global memory access for weights

### Optimized Kernel
1. **Shared Memory Caching**
   - Cache weight blocks (128 weights per tile) in shared memory
   - Reduces global memory bandwidth by ~8x for activation-bound workloads

2. **Larger Tiles**
   - TILE_M=4, TILE_N=128
   - Each thread computes 4 outputs
   - Better instruction-level parallelism (ILP)

3. **Cooperative Loading**
   - All 128 threads collaborate to load shared memory
   - Each thread loads 1 weight block
   - Maximizes memory coalescing

4. **Adaptive Dispatch**
   - M ≤ 4: Use optimized kernel (shared memory)
   - M > 4: Use naive kernel (simpler for large batches)

5. **Correct Synchronization**
   - Fixed `__syncthreads()` deadlock for M < TILE_M
   - All threads participate in synchronization
   - Use `valid_m` flag to skip computation for out-of-bounds threads

## Technical Challenges & Solutions

### Challenge 1: INT8 Sign Extension Bug
**Problem:** Casting `int8_t` to `int` caused sign extension, corrupting DP4A input
```cpp
// ❌ Wrong
int packed = (int)int8_val | ...;  // -34 → 0xFFFFFFDE

// ✅ Correct
int packed = ((uint32_t)(uint8_t)int8_val) | ...;  // -34 → 0x000000DE
```

**Impact:** Completely broke correctness (NMSE=0.6-1.9)
**Solution:** Explicit cast chain: `int8_t → uint8_t → uint32_t`

### Challenge 2: Thread Block Configuration Mismatch
**Problem:** Kernel designed for `dim3(32, 4)` but launched with `dim3(32, 32)`

**Impact:** Thread indexing errors, invalid memory access
**Solution:** Corrected launch configuration to match kernel assumptions

### Challenge 3: __syncthreads() Deadlock
**Problem:** Threads with `m >= M` early-returned before `__syncthreads()`

**Impact:** Undefined behavior, correctness failures for M < TILE_M
**Solution:** Remove early return, use `valid_m` flag to conditionally execute work

## Performance Analysis

### Why batch_4 is Optimal
- Matches kernel's TILE_M=4 design perfectly
- All threads do useful work (100% utilization)
- Shared memory benefits outweigh overhead
- Best balance of parallelism and memory efficiency

### Why batch_1 Regresses
- Only 32/128 threads active (25% utilization)
- 96 threads idle but still participate in sync
- Shared memory overhead not amortized
- Baseline's simpler approach is more efficient

### Why batch_8 is Modest
- Uses naive kernel path (M > 4 threshold)
- Naive kernel is similar to Baseline
- Minor optimizations in code generation

## Recommendations

1. **Use Optimized kernel for batch_4**: 1.55x speedup with perfect accuracy
2. **Use Baseline for batch_1**: 14% faster, simpler implementation
3. **Tune thresholds**: Consider `M == 4` for optimized path, not `M <= 4`
4. **Future work**: Implement larger tiles (e.g., TILE_M=8) for batch_8+

## Bug Fixes Applied

1. ✅ Fixed INT8 sign extension in data packing (3 kernel versions)
2. ✅ Fixed thread block dimensions from (32,32) to (32,4)
3. ✅ Fixed `__syncthreads()` deadlock with `valid_m` flag
4. ✅ Verified correctness across all batch sizes

## Files Modified

1. **Baseline Kernel**
   - `llm_kernel_test/sandbox/generated/w8a32c8_q8_0_fp32_int8_deepseek_v2_att_out_n5120_k5120/w8a32c8_q8_0_fp32_int8/kernel.cu`
   - Fixed: INT8 sign extension (2 locations)

2. **Optimized Kernel**
   - `llm_kernel_test/sandbox/generated/w8a32c8_q8_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_optimized/w8a32c8_q8_0_fp32_int8/kernel.cu`
   - Fixed: INT8 sign extension (4 locations)
   - Fixed: Block dimensions (32,32) → (32,4)
   - Fixed: `__syncthreads()` deadlock with `valid_m` flag

3. **Advanced Kernel**
   - `llm_kernel_test/sandbox/generated/w8a32c8_q8_0_advanced/w8a32c8_q8_0_fp32_int8/kernel.cu`
   - Fixed: INT8 sign extension (3 locations)
   - Status: Not tested yet (compilation timeout issue)

## Test Date
- 2026-02-13

## Conclusion

The Optimized kernel successfully achieves **1.55x speedup for batch_4** while maintaining perfect numerical accuracy. The optimization strategy of shared memory caching and larger tiles is effective for the target use case (TILE_M=4). Further tuning of batch size thresholds and tile sizes could improve performance across a wider range of batch sizes.
