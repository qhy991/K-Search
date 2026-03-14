# W4A32C8 Q4_0 DeepSeek V2 LM Head Optimization Summary

## Problem Specification

- **Operator**: Quantized GEMM (W4A32C8)
- **Quantization**: Q4_0 (4-bit weights with FP16 scales)
- **Model**: DeepSeek V2 LM Head
- **Dimensions**:
  - N = 102400 (output features)
  - K = 5120 (input features)
  - M = variable (1, 2, 3, 4, 5, 8, 512)

## Hardware Profile

- **GPU**: NVIDIA GeForce RTX 4090
- **Compute Capability**: 8.9
- **SM Count**: 128
- **Peak FP32 TFLOPS**: ~82.6
- **Peak Memory Bandwidth**: ~1.008 TB/s
- **FP32 Ridge Point**: ~82 FLOPs/Byte

## Roofline Analysis

### Operational Intensity (OI) Calculation

```
FLOPs = 2 × M × N × K = 2 × M × 102400 × 5120 = M × 1.05 × 10^9

Bytes transferred (weight not cached):
  Activation: M × 5120 × 4 = M × 20,480 bytes
  Weight: 102400 × 160 × 20 = 327,680,000 bytes
  Output: M × 102400 × 4 = M × 409,600 bytes
  Total: ~328 MB for M=1

OI (M=1) ≈ 3.2 FLOPs/Byte << Ridge Point (82)
```

**Conclusion**: The kernel is **heavily memory-bound**. Optimization should focus on:
1. Efficient memory access patterns
2. Cache utilization
3. Minimizing memory traffic

## Optimization Journey

### Version 1-2: Baseline (Shared Memory Approach)

**Strategy**: Use shared memory to cache activation rows

**Configuration**:
- 1D grid: `dim3 grid(M)`
- Block size: 256 threads
- Shared memory: 5120 floats for activation

**Performance**:
| Config | Latency | TFLOPS |
|--------|---------|--------|
| M=1    | 76.2ms  | 0.014  |
| M=8    | 72.3ms  | 0.116  |
| M=512  | 449ms   | 1.195  |

**Issues**: Poor memory coalescing for weight access patterns

### Version 3: 2D Grid Breakthrough

**Key Innovation**: Switched to 2D grid without shared memory

**Configuration**:
- 2D grid: `dim3 grid(M, N/256)`
- Block size: `dim3 block(1, 256)`
- Direct global memory access

**Performance**:
| Config | Latency | TFLOPS | Speedup |
|--------|---------|--------|---------|
| M=1    | 1.32ms  | 0.792  | **58x** |
| M=8    | 4.73ms  | 1.772  | **15x** |
| M=512  | 294ms   | 1.828  | **1.53x** |

**Why it worked**:
- Better parallelism distribution across N dimension
- L2 cache effectively handles weight reuse
- Coalesced memory access patterns
- No synchronization overhead

### Version 4: Dual Kernel Attempt

**Strategy**: Separate kernels for small/large batches

**Result**: Performed worse than v3 due to occupancy issues

### Version 5: Clean v3 Design

**Refinement**: Streamlined v3 with cleaner code structure

**Performance**:
| Config | Latency | TFLOPS |
|--------|---------|--------|
| M=1    | 1.32ms  | 0.794  |
| M=8    | 4.81ms  | 1.744  |
| M=512  | 293ms   | 1.835  |

### Version 6: Final Optimized Version

**Final optimizations**:
- `__launch_bounds__(256)` for better register allocation
- Maintained proven v3/v5 configuration

**Performance**:
| Config | Latency | TFLOPS |
|--------|---------|--------|
| M=1    | 1.32ms  | 0.792  |
| M=8    | 4.71ms  | 1.780  |
| M=512  | 293ms   | 1.834  |

## Best Version

**v6** is the best performing version with:
- Consistent performance across all batch sizes
- Clean, maintainable code structure
- Optimal occupancy and memory patterns

Location: `kernel.cu`

## Key Technical Insights

### 1. Q4_0 Format Handling

```cuda
// Q4_0 block structure: scale (FP16) + 16 packed bytes
// llama.cpp format:
//   - Positions 0-15: low nibbles of bytes 0-15
//   - Positions 16-31: high nibbles of bytes 0-15

// Unpacking formula: val = scale × (q - 8)
```

### 2. Optimal Kernel Configuration

```cuda
// 2D grid for parallelism across M and N
dim3 grid(M, (N + 255) / 256);
dim3 block(1, 256);

// Each thread computes one output element (m, n)
// No shared memory - rely on L2 cache
```

### 3. Memory Access Pattern

- **Coalesced reads**: Activation rows accessed sequentially
- **Weight reuse**: Each weight block accessed 256 times (from L2)
- **Sequential writes**: Output written with coalesced pattern

## Performance vs Theoretical Peak

The achieved 1.834 TFLOPS is ~2.2% of peak FP32 (82.6 TFLOPS), which is expected for a memory-bound kernel:

```
Effective Bandwidth = FLOPS / OI = 1.834 / 3.2 ≈ 0.57 TB/s

This is ~57% of peak bandwidth (1.008 TB/s),
indicating good memory efficiency for quantized operations.
```

## Files Included

- `kernel.cu` - Final optimized kernel (v6)
- `test_results.json` - Performance test results
- `summary.md` - This document

## Compilation

```bash
nvcc -O3 --use_fast_math -gencode arch=compute_89,code=sm_89 \
     -std=c++17 -Xcompiler -fPIC \
     kernel.cu -shared -o w4a32c8_q4_0_deepseek_v2_lm_head.so
```

## Usage

```python
import torch
from torch.utils.cpp_extension import load

# Load compiled kernel
kernel = load('w4a32c8_q4_0_deepseek_v2_lm_head',
              extra_cflags=['-O3'],
              extra_cuda_cflags=['-O3'])

# Run inference
output = kernel.forward(weight, activation, M, N, K)
```

## Conclusion

The optimization journey achieved:
- **58x speedup** for single token (M=1)
- **15x speedup** for small batch (M=8)
- **1.5x speedup** for large batch (M=512)

The key breakthrough was switching from a 1D grid with shared memory to a 2D grid with direct global memory access, leveraging the L2 cache for weight reuse while maximizing parallelism across the output dimension.
