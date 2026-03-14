# Q4_0 GEMM Kernel for DeepSeek-V2 Attention Output
## w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120

### Problem Definition
- **Operation**: Quantized GEMM (Matrix Multiplication with Q4_0 weights)
- **Dimensions**: M (batch), N=5120 (output), K=5120 (input)
- **Weight Format**: Q4_0 (4-bit quantization with block size 32)
- **Activation**: FP32 dynamically quantized during compute
- **Output**: FP32

### Hardware Profile
- **GPU**: NVIDIA GeForce RTX 4090
- **Compute Capability**: 8.9
- **SM Count**: 128
- **Peak FP32**: 82.6 TFLOPS
- **Peak Bandwidth**: 1008 GB/s

### Roofline Analysis
All configurations are **compute-bound** (OI >> 0.1 ridge point):

| M | OI (FLOPs/Byte) | Bottleneck | Strategy |
|---|-----------------|------------|----------|
| 1 | 3.2 | Compute | ILP, loop unrolling |
| 8 | 25.1 | Compute | Vectorized operations |
| 512 | 718.6 | Compute | Maximize throughput |

### Q4_0 Format Details
- **Block Size**: 32 values
- **Storage**: 18 bytes per block
  - 2 bytes: FP16 scale (delta)
  - 16 bytes: 32 x 4-bit values (packed)
- **Encoding**: Offset-8: `q = round(val / scale + 8)`
- **Decoding**: `val = scale * (q - 8)`
- **Packing**: llama.cpp format
  - Positions 0-15: Low nibbles of bytes 0-15
  - Positions 16-31: High nibbles of bytes 0-15

### Performance Results

#### Best Version: v8 (Hybrid Strategy)

| Configuration | Latency | TFLOPS | % Peak | Status |
|--------------|---------|--------|--------|--------|
| M=1 (single_token) | 0.202 ms | 0.26 | 0.31% | ✅ |
| M=8 (small_batch) | 0.328 ms | 1.28 | 1.55% | ✅ |
| M=512 (large_batch) | 17.886 ms | 1.50 | 1.82% | ✅ |

#### Version Comparison

| Version | M=1 TFLOPS | M=8 TFLOPS | M=512 TFLOPS |
|---------|-----------|------------|-------------|
| v3 | 0.26 | 1.05 | 1.247 |
| v6 | 0.26 | 1.05 | 1.247 |
| v7 | 0.04 | 1.28 | 1.50 |
| **v8** | **0.26** | **1.28** | **1.50** |

### Kernel Strategy (v8)

The hybrid approach selects the optimal kernel based on batch size:

1. **M=1 (Single Token)**: Simple thread-per-output
   - 256 threads per block
   - Each thread computes one output element
   - Minimizes kernel launch overhead

2. **2 ≤ M < 8 (Medium Batch)**: 2 outputs per thread
   - 128 threads per block (256 outputs per block)
   - Each thread computes 2 outputs for ILP
   - Better warp utilization

3. **M ≥ 8 (Large Batch)**: 8 outputs per block
   - 8 threads per block
   - Optimized for high occupancy
   - Better memory coalescing

### Key Optimizations

1. **Instruction-Level Parallelism (ILP)**
   - Loop unrolling: `#pragma unroll 4`
   - Multiple outputs per thread for medium batches

2. **Memory Access**
   - Coalesced global memory reads
   - Efficient FP16 to FP32 conversion

3. **Thread Block Configuration**
   - Adaptive based on batch size
   - Optimized for GPU occupancy

### Correctness
All configurations pass with **NMSE = 0.0** (within 0.05 threshold).

### Files
```
attempts/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_v8/
├── kernel.cu          # Best hybrid kernel implementation
└── test_results.json  # Performance and correctness results
```

### Usage
```python
import torch
from torch.utils.cpp_extension import load

# Load the compiled kernel
kernel = load(
    name="w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_quant_gemm_test",
    sources=["attempts/w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120_v8/kernel.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

# Run kernel
output = kernel.forward(weight, activation, M, N, K)
```

### Future Optimizations
1. Tensor Core utilization (needs INT8 tensor cores)
2. Shared memory tiling for weight reuse
3. Asynchronous copy for overlapping compute and memory
4. Multi-kernel pipeline for very large batches
