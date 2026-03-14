# Quantized GEMM: w4a32c8_q4_0_fp32_int8_ds3_moe_routing_up_n7168_k2048

## Task Summary
- **Operator**: W4A32C8 Quantized GEMM (Q4_0 weights, FP32 activation with Q8_1 dynamic quantization)
- **Shape**: M=var, N=7168, K=2048
- **Model**: DeepSeek-V3 MoE Routing Expert Up projection
- **GPU**: NVIDIA GeForce RTX 4090 (Compute Capability 8.9)

## Roofline Analysis

### Hardware Specs (RTX 4090)
- SM Count: 128
- Peak FP32 Performance: ~82.6 TFLOPS
- Memory Bandwidth: ~1.008 TB/s
- Ridge Point: 81.9 FLOPs/Byte

### Operational Intensity by Batch Size
| M  | FLOPs       | Bytes       | OI (FLOPs/Byte) | Regime     |
|----|-------------|-------------|-------------------|------------|
| 1  | 29.36M      | 8.64M       | 3.4               | Memory     |
| 2  | 58.72M      | 17.0M       | 3.5               | Memory     |
| 4  | 117.4M      | 33.6M       | 3.5               | Memory     |
| 8  | 234.9M      | 66.5M       | 3.5               | Memory     |
| 512| 15.03B      | 27.1M       | 555               | Compute    |

**Key Finding**: Small M is heavily memory-bound (OI << 81.9), large M is compute-bound (OI >> 81.9).

## Performance Results by Version

| Version | single_token | small_batch | large_batch | Notes |
|---------|--------------|-------------|-------------|-------|
| v1      | 1.26 TFLOPS  | 1.57 TFLOPS | 1.20 TFLOPS | Strategy dispatch (M<=8, M>64, large SM tiling) |
| v2      | 1.27 TFLOPS  | 1.57 TFLOPS | 0.56 TFLOPS | Large batch degraded - strided access pattern |
| v3      | 1.27 TFLOPS  | 1.57 TFLOPS | 1.22 TFLOPS | Fixed shared memory loading |
| v4      | 1.22 TFLOPS  | 1.57 TFLOPS | 1.14 TFLOPS | Direct 1D thread approach |
| v5      | 1.26 TFLOPS  | 0.84 TFLOPS | 0.55 TFLOPS | Large tile size (128) caused inefficiency |
| v6      | 1.26 TFLOPS  | 0.84 TFLOPS | 0.54 TFLOPS | M<=16 threshold caused small_batch degradation |
| v7      | 1.27 TFLOPS  | 1.57 TFLOPS | 1.22 TFLOPS | **BEST** - Same as v3, M<=8 threshold |

## Final Performance

| Config       | M   | Latency (ms) | TFLOPS | NMSE    |
|--------------|-----|--------------|---------|---------|
| single_token | 1   | 0.023        | 1.274   | 0.00005 |
| small_batch  | avg | 0.150        | 1.569   | 0.00016 |
| large_batch  | 512 | 12.321       | 1.220   | 0.00022 |

## Strategy Used

### Small Batch (M <= 8): Memory-Bound Warp-Centric Kernel
- Each warp processes multiple N outputs
- Threads split K work across the warp
- Minimizes global memory access through warp reduction

### Large Batch (M > 8): Shared Memory Tiling Kernel
- 2D tiling: TILE_N=64, TILE_M=4
- Each thread computes 2 N outputs for ILP
- Activation loaded into shared memory once per K block
- Reduces activation memory traffic by TILE_N factor

## Key Optimizations

1. **dp4a instruction**: Uses `__dp4a` for efficient int8 dot product on CC >= 6.1
2. **Dynamic activation quantization**: Per-block (32 elements) Q8_1 style
3. **Strategy dispatch**: Different kernels for different M regimes
4. **Shared memory tiling**: For compute-bound regime
5. **Instruction-level parallelism**: Multiple accumulators per thread

## Limitations

The large batch performance (~1.2 TFLOPS) is limited by:
1. **Weight memory bandwidth**: N=7168 with 8.2 MB weights means each output element requires accessing different weight blocks
2. **Quantized format overhead**: Q4_0 requires per-block scale reading and unpacking
3. **No tensor cores**: This format doesn't leverage RTX 4090's tensor cores efficiently

## Reference Comparison

Compared to ds2_moe_routing_up_n5120_k1536 (N=5120, K=1536) which achieved 3.0 TFLOPS on large batch:
- Our N is 40% larger (7168 vs 5120) = 40% more weight data
- Our K is 33% larger (2048 vs 1536) = 33% more FLOPs
- Weight size: 8.2 MB vs 4.4 MB (87% increase)

The larger weight size significantly increases memory bandwidth pressure.
