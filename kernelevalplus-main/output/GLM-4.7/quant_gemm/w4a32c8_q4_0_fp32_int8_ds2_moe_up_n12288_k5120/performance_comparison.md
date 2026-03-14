# Performance Comparison - W4A32C8 Q4_0 Quantized GEMM

## Test Results Summary

| Version | M=1 (TFLOPS) | M=512 (TFLOPS) | Correctness | Notes |
|---------|--------------|----------------|-------------|-------|
| v1 | ❌ Failed | ❌ Failed | NMSE > 1.8 | Wrong unpacking |
| v2 | 0.441 | 0.886 | ✅ NMSE=0.00003 | First working version |
| v3 | 0.273 | 1.944 | ✅ NMSE=0.00003 | Large batch optimized |
| v4 | 0.472 | **1.942** | ✅ NMSE=0.00003 | Strategy dispatch |
| v5 | 0.472 | 0.700 | ✅ NMSE=0.00003 | Warp-level (failed) |
| v6 | 0.029 | 0.333 | ✅ NMSE=0.00003 | Shared mem (failed) |
| v7 | 0.135 | 1.890 | ✅ NMSE=0.00003 | Medium optimization |
| v8 | 0.472 | 1.941 | ✅ NMSE=0.00003 | Micro-optimizations |
| **v9** | **0.569** | 1.307 | ✅ NMSE=0.00003 | **DP4A optimized** ⭐ |
| v10 | 0.264 | 1.941 | ✅ NMSE=0.00003 | Hybrid attempt |
| v11 | 0.263 | 1.936 | ✅ NMSE=0.00003 | Hybrid v2 |

## Key Findings

### Best for Small Batch (M ≤ 8): Version 9 (DP4A)
- **Performance**: 0.569 TFLOPS for M=1
- **Improvement**: +29% over v2, +21% over v4
- **Key Innovation**: Uses `__dp4a` instruction for vectorized INT8 dot product

### Best for Large Batch (M > 128): Version 4 (Strategy Dispatch)
- **Performance**: 1.942 TFLOPS for M=512
- **Approach**: Multiple kernels optimized for different batch size ranges

## Performance Breakdown by Batch Size

| M (Batch Size) | v4 TFLOPS | v9 TFLOPS | Best Version |
|----------------|-----------|-----------|--------------|
| 1 | 0.472 | **0.569** | v9 (DP4A) |
| 2-4 | ~0.9-1.0 | **~1.3-1.5** | v9 (DP4A) |
| 8-32 | ~1.0-1.1 | **~1.7** | v9 (DP4A) |
| 128 | ~1.9 | 1.3 | v4 (Strategy) |
| 512 | **1.942** | 1.307 | v4 (Strategy) |

## Latency Comparison

| Configuration | v4 Latency | v9 Latency | Best |
|--------------|------------|------------|------|
| M=1 | 0.267 ms | **0.221 ms** | v9 |
| M=5 (small) | 0.942 ms | **0.582 ms** | v9 |
| M=512 (large) | **33.2 ms** | 49.3 ms | v4 |

## Recommendation

**For LLM Inference (typical batch sizes 1-8)**:
```bash
# Use v9 DP4A-optimized kernel
kernel_dp4a_small_batch.cu
```

**For Batch Processing (batch sizes 128+)**:
```bash
# Use v4 strategy dispatch kernel
kernel_strategy_dispatch.cu
```
