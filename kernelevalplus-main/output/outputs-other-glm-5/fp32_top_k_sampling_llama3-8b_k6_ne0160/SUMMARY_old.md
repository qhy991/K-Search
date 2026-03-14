# Top-K Sampling Kernel Optimization Summary

## Task Definition

- **Operator**: Top-K Sampling
- **Model**: Llama3-8B
- **Parameters**: K=6, vocab_subset=160
- **Input**: probs [batch_size, 160] FP32
- **Output**: samples [batch_size] int64
- **Formula**: `top_indices = argsort(probs, descending=True)[:6]; samples = categorical(probs[top_indices])`

## Hardware Environment

- **GPU**: NVIDIA GeForce RTX 4090
- **Compute Capability**: 8.9
- **SM Count**: 128
- **Peak Memory Bandwidth**: 1008 GB/s

## Roofline Analysis

| Metric | Value |
|--------|-------|
| Memory Access | batch_size × 160 × 4 = 640 × batch_size bytes |
| Computation | ~2880 × batch_size FLOPs |
| Operational Intensity | ~4.5 FLOPs/Byte |
| Ridge Point | 81.9 FLOPs/Byte |
| **Bottleneck** | **Memory-Bound** (OI << Ridge Point) |

## Optimization Journey

### Version 1: Baseline Implementation

**Strategy**: One warp per batch element, shared memory for top-k heap

**Results**:
| Test | Latency | Bandwidth | Baseline Ratio |
|------|---------|-----------|----------------|
| batch_1 | 0.012 ms | 0.06 GB/s | 54.5% |
| batch_8 | 0.013 ms | 0.38 GB/s | 66.7% |
| batch_512 | 0.013 ms | 26.1 GB/s | 91.3% |

**Issues**:
- Too few threads (32) per block
- Heavy serialization through thread 0
- Large shared memory overhead

---

### Version 2: Increased Parallelism

**Optimizations**:
- Increased threads per block to 256
- Warp-level primitives for reduction
- Reduced shared memory usage

**Results**:
| Test | Latency | Bandwidth | Baseline Ratio |
|------|---------|-----------|----------------|
| batch_1 | 0.007 ms | 0.09 GB/s | 81.8% |
| batch_8 | 0.008 ms | 0.68 GB/s | 119.3% |
| batch_512 | 0.025 ms | 12.9 GB/s | 45.1% |

**Issues**:
- Improved small batch performance
- Large batch performance regressed

---

### Version 3: Strategy Dispatch

**Optimizations**:
- Separate kernels for different batch sizes
- Small batch (≤32): One warp per batch
- Large batch (>32): One block per batch with 128 threads

**Results**:
| Test | Latency | Bandwidth | Baseline Ratio |
|------|---------|-----------|----------------|
| batch_1 | 0.007 ms | 0.09 GB/s | 81.8% |
| batch_8 | 0.007 ms | 0.72 GB/s | 126.3% |
| batch_512 | 0.008 ms | 40.0 GB/s | 140.0% |

**Improvements**:
- Excellent large batch performance
- Small batch still below baseline

---

### Version 4 (Final): Vectorized Loads

**Optimizations**:
- Added specialized single-batch kernel with float4 vectorized loads
- Launch bounds optimization
- Fully unrolled warp reduction

**Results**:
| Test | Latency | Bandwidth | Baseline Ratio |
|------|---------|-----------|----------------|
| batch_1 | 0.006 ms | 0.10 GB/s | 90.9% |
| batch_8 | 0.007 ms | 0.75 GB/s | 131.6% |
| batch_512 | 0.008 ms | 39.6 GB/s | 138.6% |

---

### Final Version: Refined Dispatch

**Final optimizations**:
- Tuned thread block configurations
- Optimized warp reduction

**Results**:
| Test | Latency | Bandwidth | Baseline | Ratio |
|------|---------|-----------|----------|-------|
| batch_1 | 0.006 ms | **0.11 GB/s** | 0.11 GB/s | **100.0%** |
| batch_8 | 0.006 ms | **0.88 GB/s** | 0.57 GB/s | **154.4%** |
| batch_512 | 0.008 ms | **39.85 GB/s** | 28.58 GB/s | **139.5%** |

## Key Optimization Techniques

### 1. Strategy Dispatch
```cpp
if (batch_size == 1) {
    // Single warp with vectorized loads
    topk_single_kernel<<<1, 32>>>(...);
} else if (batch_size <= 32) {
    // 4 warps per block, one warp per batch
    dim3 block(32, 4);
    topk_small_kernel<<<...>>>(...);
} else {
    // One block per batch, 128 threads
    topk_large_kernel<<<batch_size, 128>>>(...);
}
```

### 2. Register-based TopK (K=6)
```cpp
struct TopK {
    float v0, v1, v2, v3, v4, v5;
    int i0, i1, i2, i3, i4, i5;

    __device__ __forceinline__ void insert(float val, int idx) {
        // Fully unrolled insertion sort
        if (val > v0) { /* shift and insert */ }
        else if (val > v1) { /* ... */ }
        // ...
    }
};
```

### 3. Warp-level Reduction
```cpp
// Warp shuffle for efficient top-k merge
for (int offset = 16; offset > 0; offset >>= 1) {
    if (lane_id < offset) {
        local.insert(__shfl_down_sync(0xffffffff, local.v0, offset), ...);
        // ... merge other top-k values
    }
}
```

### 4. Vectorized Memory Access
```cpp
// float4 loads for coalesced access
const float4* probs_vec = reinterpret_cast<const float4*>(probs);
float4 vec = probs_vec[v];
local.insert(vec.x, base);
local.insert(vec.y, base + 1);
// ...
```

## Final Performance Summary

| Metric | batch_1 | batch_8 | batch_512 |
|--------|---------|---------|-----------|
| Latency (ms) | 0.006 | 0.006 | 0.008 |
| Bandwidth (GB/s) | 0.11 | 0.88 | 39.85 |
| Baseline (GB/s) | 0.11 | 0.57 | 28.58 |
| **Ratio** | **100%** | **154%** | **140%** |

## Files

```
experiments/fp32_top_k_sampling_llama3_8b_k6_ne0160/
├── kernel_best.cu          # Best performing kernel
├── SUMMARY.md              # This summary
├── versions/
│   ├── kernel_v1.cu        # Initial implementation
│   ├── kernel_v2.cu        # Increased parallelism
│   ├── kernel_v3.cu        # Strategy dispatch
│   └── kernel_v4.cu        # Vectorized loads
└── test_results/
    ├── test_v1_results.json
    ├── test_v2_results.json
    ├── test_v3_results.json
    ├── test_v4_results.json
    └── test_final_results.json
```

## Conclusion

The final kernel achieves:
- **100% of baseline** for batch_1
- **154% of baseline** for batch_8
- **140% of baseline** for batch_512

Key success factors:
1. **Strategy dispatch** - Different kernels optimized for different batch sizes
2. **Register-based TopK** - Avoids memory traffic for small K
3. **Warp shuffle** - Efficient reduction without shared memory
4. **Vectorized loads** - Better memory bandwidth utilization
