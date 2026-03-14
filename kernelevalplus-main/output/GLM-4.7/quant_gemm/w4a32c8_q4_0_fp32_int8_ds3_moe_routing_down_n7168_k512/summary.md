# w4a32c8_q4_0_fp32_int8_ds3_moe_routing_down_n7168_k512 - Kernel Development Summary

## Task Definition
- **Operator**: Quantized GEMM with BLOCK_Q4_0 weights and FP32 activations
- **Dimensions**: N=7168 (output), K=512 (input), M (variable batch)
- **Format**: llama.cpp BLOCK_Q4_0 pattern
- **Quantization**: 4-bit weights (per-block of 32 values), FP32 activations dynamically quantized to Q8_1 style

## Hardware Profile
- **GPU**: NVIDIA GeForce RTX 4090
- **Compute Capability**: 8.9
- **SM Count**: 128
- **Memory**: 23.6 GB

## Roofline Analysis
| M   | OI (FLOPs/Byte) | Ridge Point | Regime        |
|-----|-------------------|-------------|---------------|
| 1   | 1.9               | 81.9        | Memory-bound   |
| 8   | 15.0              | 81.9        | Memory-bound   |
| 512 | 759.5             | 81.9        | Compute-bound  |

## Performance Results

### Final Version (attempt/final)
| Config      | M    | N    | K    | Latency (ms) | GFLOPS |
|-------------|------|------|------|--------------|--------|
| single_token| 1    | 7168 | 512  | 0.024        | 306.7  |
| small_batch | 8    | 7168 | 512  | 0.070        | 841.4  |
| large_batch | 512  | 7168 | 512  | 3.059        | 1228.4 |

### Correctness
- **NMSE**: 0.000029 (well below threshold of 0.05)
- All test configurations pass

## Implementation Details

### Strategy Dispatch
The kernel uses 4 different strategies based on batch size:

1. **Small M (M <= 4)**: One thread per output, optimized for memory bandwidth
2. **Medium M (4 < M <= 32)**: Multiple outputs per thread (N_PER_THREAD=4)
3. **Large M (32 < M <= 128)**: Process multiple M rows (M_PER_THREAD=2, N_PER_THREAD=8)
4. **Very Large M (M > 128)**: Maximize throughput (M_PER_THREAD=4, N_PER_THREAD=4)

### Key Optimizations
- Per-block activation quantization (Q8_1 style)
- BLOCK_Q4_0 compatible unpacking with llama.cpp ordering
- Unrolled loops for critical path
- Multiple outputs per thread for larger batches

## Comparison with Baseline
The closest baseline found was for M=7168, N=1, K=512 achieving ~1080 GFLOPS.
Our kernel's performance is reasonable given the different memory access patterns.

## Files
- **Final Kernel**: `attempts/final/kernel.cu`
- **Test Results**: `attempts/final/test_results.json`
