# W4A32C8 Q4_0 × FP32 Quantized GEMM - Best Version

## Quick Reference

**Kernel**: `kernel_best.cu`
**Target**: DeepSeek-V2 MoE Routing Down Projection (N=1536, K=5120)
**Best Version**: v8

## Performance (RTX 4090)

| Config | M | Latency (ms) | TFLOPS |
|--------|---|--------------|--------|
| single_token | 1 | 0.204 | 0.077 |
| small_batch | 2 | 0.203 | 0.619 |
| large_batch | 512 | 4.240 | 1.899 |

## Files

- `kernel_best.cu` - Best performing kernel (v8)
- `summary.md` - Detailed optimization journey
- `test_results.json` - Complete test results

## Usage

```python
import torch

# Load the compiled extension
kernel = torch.ops.load_library("path/to/compiled.so")

# Call the kernel
output = kernel.forward(weight, activation, M, N, K)
```

## Key Optimizations in v8

1. **4-block chunking**: Load 4 activation blocks per shared memory access
2. **Reduced sync**: Process multiple blocks per synchronization
3. **Shared memory**: Cache activation values reused across threads
4. **Proper alignment**: Avoid unsafe vectorized loads

## Correctness

✅ NMSE = 0.0 (exact match with reference)
