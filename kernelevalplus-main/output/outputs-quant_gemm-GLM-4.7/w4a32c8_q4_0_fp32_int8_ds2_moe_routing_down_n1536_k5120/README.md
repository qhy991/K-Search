# W4A32C8 Q4_0 Quantized GEMM - Best Implementation

## Overview

This is the best performing implementation of the W4A32C8 Q4_0 quantized GEMM kernel for DeepSeek-V2 MoE Routing Down projection (N=1536, K=5120).

## Performance (RTX 4090)

| Batch Size | TFLOPS | Latency (ms) | Regime |
|------------|--------|--------------|---------|
| M=1        | 0.85   | 0.018        | Memory-bound |
| M=8        | 2.03   | 0.062        | Transition |
| M=512      | 2.62   | 3.08         | Compute-bound |

## Usage

```python
import torch

# Load the compiled kernel
kernel = torch.ops.w4a32c8_q4_0_fp32_int8_ds2_moe_routing_down_n1536_k5120_quant_gemm_test

# Run inference
output = kernel.forward(weight_q4_0, activation_fp32, M, N, K)
```

## Arguments

- `weight_q4_0` (torch.Tensor, uint8): Q4_0 quantized weights, shape [N, K/32 * 18]
- `activation_fp32` (torch.Tensor, float32): FP32 activations, shape [M, K]
- `M` (int): Batch dimension
- `N` (int): Output dimension (1536 for this layer)
- `K` (int): Input dimension (5120 for this layer)

## Output

- `output` (torch.Tensor, float32): Output tensor, shape [M, N]

## Compilation

```bash
nvcc -O3 --use_fast_math -std=c++17 \
    -arch=compute_89 -code=sm_89 \
    -shared -Xcompiler "-fPIC" \
    $(python3 -m torch.utils.cpp_extension \
        --python-cpp-ext \
        --cuda-cpp-ext) \
    kernel.cu -o w4a32c8_q4_0_fp32_int8_ds2_moe_routing_down_n1536_k5120.so
```

## Implementation Details

### Q4_0 Format

- **Block size**: 32 values per block
- **Bytes per block**: 18 bytes
  - Bytes 0-1: FP16 scale (d_w)
  - Bytes 2-17: Packed 4-bit values (16 bytes)

### Unpacking (llama.cpp compatible)

```cuda
// Lane 0-15: low nibbles (positions 0-15)
q = (packed[lane_id] & 0x0F);

// Lane 16-31: high nibbles (positions 16-31)
q = ((packed[lane_id - 16] >> 4) & 0x0F);
```

### Dequantization

```cuda
float w = a * d_w * static_cast<float>(q - 8);
```

### Kernel Design

- **Warp-level collaboration**: 32 lanes compute one output element together
- **Loop unrolling**: 4x unrolling for better ILP
- **Warp reduction**: `__shfl_down_sync` for efficient partial sum aggregation
- **128 threads/block**: 4 warps per block, 4N values processed per block

## Correctness

✅ All tests pass with NMSE = 0.0

## Test Results

See `test_results.json` for detailed performance and correctness metrics.
