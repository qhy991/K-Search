# Version Comparison - W4A32C8 Q4_0 Quantized GEMM

## All Versions Performance Data

| Version | M=1 (TFLOPS) | M=8 (TFLOPS) | M=512 (TFLOPS) | Correctness | Key Innovation |
|---------|-------------|--------------|----------------|-------------|----------------|
| v1 | FAIL | FAIL | FAIL | ❌ NMSE=2.67 | Initial attempt (wrong formula) |
| v2 | 0.650 | 1.747 | **2.000** | ✅ NMSE=0.00 | Baseline simple implementation |
| v3 | 0.108 | 0.871 | 1.218 | ✅ NMSE=0.00 | Multiple N per thread |
| v4 | FAIL | FAIL | PASS | ❌ NMSE=1.00 | Warp reduction bug |
| v5 | 0.653 | **2.027** | 1.874 | ✅ NMSE=0.00 | Vectorized float4 loads |
| v6 | **1.796** | 0.619 | 1.985 | ✅ NMSE=0.00 | Warp + shared activation cache |
| v7 | 0.599 | 1.747 | 1.835 | ✅ NMSE=0.00 | Simplified design |
| v8 | **1.798** | 1.908 | **2.002** | ✅ NMSE=0.00 | **Combined best kernels** |

## Detailed Analysis by Version

### v1 - FAILED
- **Issue**: Incorrect computation formula
- **NMSE**: 2.67 (single_token), 14.47 (small_batch), 8.08 (large_batch)
- **Cause**: Misunderstanding of Q4_0 format and computation pattern

### v2 - PASS (Baseline)
- **Approach**: Simple thread-per-output with shared memory
- **Performance**:
  - M=1: 0.65 TFLOPS
  - M=8: 1.75 TFLOPS
  - M=512: 2.00 TFLOPS (best for large M)
- **Strengths**: Simple, correct, works well for large M
- **Weaknesses**: Suboptimal for small M

### v3 - PASS
- **Approach**: Multiple N values per thread
- **Performance**: Worse than v2 across all configs
- **Lesson**: Multiple-N-per-thread hurts coalescing for this access pattern

### v4 - FAILED
- **Issue**: Warp reduction logic error
- **NMSE**: 1.00 (single_token, small_batch)
- **Lesson**: Warp reduction requires all lanes to compute partial sums

### v5 - PASS
- **Approach**: Vectorized float4 loads
- **Performance**:
  - M=1: 0.65 TFLOPS
  - M=8: **2.03 TFLOPS** (best for small batch)
  - M=512: 1.87 TFLOPS
- **Innovation**: Memory coalescing via float4 loads
- **Best for**: M=8 (small batch)

### v6 - PASS
- **Approach**: Warp-level + shared activation cache
- **Performance**:
  - M=1: **1.80 TFLOPS** (best for single token, 3x improvement)
  - M=8: 0.62 TFLOPS (regression)
  - M=512: 1.99 TFLOPS
- **Innovation**:
  - Cache entire activation in shared memory (20KB)
  - Each warp computes 1 output
  - Lane-strided K blocks
- **Best for**: M=1 (single token)

### v7 - PASS
- **Approach**: Simplified thread-per-output
- **Performance**: Similar to v2
- **Lesson**: Simplicity is valuable, but v6's warp approach is better for M=1

### v8 - PASS (FINAL)
- **Approach**: Adaptive kernel selection
- **Performance**:
  - M=1: **1.80 TFLOPS** (from v6)
  - M=8: 1.91 TFLOPS (balanced for M=2-8)
  - M=512: **2.00 TFLOPS** (from v2)
- **Strategy**:
  - M=1: Use v6's warp + shared cache
  - M≤8: Use v5's vectorized loads
  - M>8: Use v2's shared memory
- **Best for**: All configurations (combined optimal)

## Performance Bank

Per M value, the best version is:

| M | Best Version | TFLOPS | Kernel Type |
|---|-------------|--------|-------------|
| 1 | v6, v8 | 1.80 | Warp + shared cache |
| 8 | v5 | 2.03 | Vectorized loads |
| 512 | v2, v8 | 2.00 | Shared memory |

## Key Insights

1. **Memory-bound regimes require different strategies**:
   - M=1: Cache activation, process many outputs
   - M=8: Vectorize loads
   - M=512: Shared memory tiles

2. **Warp-level processing helps for M=1**:
   - Caching 20KB activation in shared memory
   - Each warp computes 1 output
   - 3x improvement over simple approach

3. **No single strategy is optimal for all M**:
   - Different access patterns favor different optimizations
   - Adaptive dispatch (v8) achieves balanced performance

4. **Baseline gap**:
   - M=1: 21% of baseline (8.54 TFLOPS)
   - Likely due to DP4A/tensor core optimizations in baseline
