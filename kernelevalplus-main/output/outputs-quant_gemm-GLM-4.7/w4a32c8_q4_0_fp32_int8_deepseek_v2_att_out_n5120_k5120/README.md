# Q4_0 GEMM Kernel - Best Version (v8)

## Quick Summary

**Operation**: Quantized GEMM for DeepSeek-V2 Attention Output  
**Best Version**: v8 (Hybrid Strategy)  
**Status**: ✅ Correct (NMSE=0) + Optimized Performance

## Performance

| Configuration | Latency | TFLOPS | vs Baseline |
|--------------|---------|--------|-------------|
| M=1 (single_token) | 0.202 ms | 0.26 | Baseline |
| M=8 (small_batch) | 0.328 ms | 1.28 | **+22%** |
| M=512 (large_batch) | 17.886 ms | 1.50 | **+20%** |

## Files

- `kernel.cu` - Best implementation (v8 hybrid strategy)
- `test_results.json` - Detailed test results
- `summary.md` - Complete optimization history

## Key Features

1. **Hybrid Strategy**: Different kernels for different batch sizes
   - M=1: Simple thread-per-output
   - 2≤M<8: 2 outputs per thread (ILP)
   - M≥8: 8 outputs per block (throughput)

2. **Optimizations**:
   - Loop unrolling (`#pragma unroll 4`)
   - Multi-output processing
   - Efficient FP16→FP32 conversion

3. **Correctness**: All tests pass with NMSE=0.0

## Usage

```cpp
// Compile
nvcc -O3 -arch=sm_89 -std=c++17 --use_fast_math \
     kernel.cu -shared -fPIC -o q4_0_gemm.so

// Run
torch::Tensor output = kernel.forward(weight, activation, M, N, K);
```

## Hardware

- Tested on: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Should work on: Any GPU with CC ≥ 7.0

## Notes

- Q4_0 format: 18 bytes per 32 values (2-byte FP16 scale + 16 bytes packed)
- Dequantization: `val = scale * (q - 8)` (offset-8 encoding)
- llama.cpp compatible packing format
