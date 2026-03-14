# Performance Comparison - All Versions

## Version-by-Version Performance

| Version | single_token (M=1) | small_batch (M≈8) | large_batch (M=512) |
|---------|-------------------|-------------------|---------------------|
| v1 (baseline) | 76.219ms, 0.014 TFLOPS | 72.261ms, 0.116 TFLOPS | 449.127ms, 1.195 TFLOPS |
| v2 (same as v1) | 76.250ms, 0.014 TFLOPS | 72.318ms, 0.116 TFLOPS | 449.171ms, 1.195 TFLOPS |
| v3 (2D grid) | 1.324ms, 0.792 TFLOPS | 4.733ms, 1.772 TFLOPS | 293.631ms, 1.828 TFLOPS |
| v4 (dual kernel) | 1.358ms, 0.772 TFLOPS | 6.992ms, 1.200 TFLOPS | 407.666ms, 1.317 TFLOPS |
| v5 (clean v3) | 1.321ms, 0.794 TFLOPS | 4.811ms, 1.744 TFLOPS | 292.580ms, 1.835 TFLOPS |
| **v6 (final)** | **1.324ms, 0.792 TFLOPS** | **4.712ms, 1.780 TFLOPS** | **292.696ms, 1.834 TFLOPS** |

## Speedup vs Baseline (v1)

| Config | v1 → v6 Speedup | TFLOPS Improvement |
|--------|-----------------|-------------------|
| M=1    | **57.6x** | 0.014 → 0.792 |
| M=8    | **15.3x** | 0.116 → 1.780 |
| M=512  | **1.53x** | 1.195 → 1.834 |

## Key Improvements by Version

### v1 → v3: The Breakthrough
- **58x faster** for single token
- **15x faster** for small batch
- Change: 1D grid → 2D grid, removed shared memory

### v3 → v6: Fine-tuning
- Minor improvements (<1%)
- Focus on code cleanliness and stability
- Added `__launch_bounds__` for better register allocation

## Best Configuration

**v6** is selected as the final version because:
- Highest TFLOPS for small_batch (1.780)
- Competitive TFLOPS for all configurations
- Clean, maintainable code
- Stable across all batch sizes

## Hardware Efficiency

```
Peak FP32 (RTX 4090): ~82.6 TFLOPS
Achieved (v6): 1.834 TFLOPS

Efficiency: 1.834 / 82.6 = 2.2%

This is excellent for a memory-bound quantized GEMM:
- OI ≈ 3.2 FLOPs/Byte
- Ridge Point ≈ 82 FLOPs/Byte
- Kernel is 25x below ridge (heavily memory-bound)
```
