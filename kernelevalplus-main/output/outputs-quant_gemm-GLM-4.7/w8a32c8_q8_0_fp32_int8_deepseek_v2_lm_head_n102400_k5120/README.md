# W8A32C8 Q8_0 Quantized GEMM - DeepSeek-V2 LM Head

**High-performance CUDA kernel for quantized matrix multiplication**

## Quick Results

| Configuration | TFLOPS | Latency (ms) | vs Reference |
|--------------|--------|--------------|--------------|
| M=1  | **1.645** | 0.637 | ✅ +0.9% |
| M=8  | 0.497 | 16.875 | 99.2% |
| M=512 | 25.758 | 20.843 | 95.5% |

## Usage

```python
import torch
from torch.utils.cpp_extension import load

# Load the compiled kernel
kernel = load('w8a32c8_q8_0_fp32_int8_deepseek_v2_lm_head_n102400_k5120')

# Prepare tensors
weight = torch.randint(0, 255, (N, K//32 * 34), dtype=torch.uint8, device='cuda')  # Q8_0
activation = torch.randn(M, K, dtype=torch.float32, device='cuda')

# Run kernel
output = kernel.forward(weight, activation, M, N, K)
```

## Compilation

```bash
nvcc -O3 --use_fast_math -std=c++17 \
    -gencode=arch=compute_89,code=sm_89 \
    -shared -Xcompiler -fPIC \
    kernel_best.cu -o w8a32c8_q8_0.so
```

## Files

| File | Description |
|------|-------------|
| `kernel_best.cu` | **Best performing kernel** (use this!) |
| `summary.md` | Complete optimization journey |
| `PERFORMANCE_COMPARISON.md` | Performance comparison chart |
| `test_results/v8_results.json` | Detailed test results |

## Key Features

- **DP4A Instruction**: Fast INT8 dot product
- **Dynamic Quantization**: On-the-fly activation quantization
- **Strategy Dispatch**: Optimal kernel selection per batch size
- **Tensor Cores**: Leverages FP16 matmul for large batches

## Technical Details

- **Problem**: Q8_0 × FP32 GEMM
- **Dimensions**: N=102400, K=5120, M∈{1,2,3,4,5,8,512}
- **Hardware**: NVIDIA RTX 4090 (Compute 8.9)
- **Correctness**: NMSE = 0.000029

## Optimization Strategy

```
if (M == 1)          → 32-thread warp with DP4A
else if (M <= 16)    → 256-thread block with DP4A
else                 → FP16 Tensor Core via PyTorch
```

---

*Generated: 2025-03-11*
*Tested on: NVIDIA GeForce RTX 4090*
