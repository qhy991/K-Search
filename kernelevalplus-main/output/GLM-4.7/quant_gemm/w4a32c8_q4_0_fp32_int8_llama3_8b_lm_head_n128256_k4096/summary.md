# W4A32C8 Q4_0 GEMM - LLaMA3-8B LM Head (N=128256, K=4096)

## Task Overview
- **Operator**: Quantized GEMM with W4A32C8 format
- **Weight quantization**: Q4_0 (4-bit weights)
- **Activation storage**: FP32 with dynamic Q8_1 per-block quantization during compute
- **Dimensions**: N=128256 (output), K=4096 (input), M variable (batch)
- **Formula**: `result = d4_0 * (d8_1 * sumi - 8 * s8_1)`

## Hardware Profile
- **GPU**: NVIDIA GeForce RTX 4090
- **Compute Capability**: 8.9
- **SM Count**: 128
- **Peak FP32 TFLOPS**: ~82.6 TFLOPS

## Roofline Analysis
Operational Intensity (OI) calculation showed the operation is compute-bound for all M values tested (OI > 3 FLOPs/Byte vs ridge ~0.1 FLOPs/Byte). However, memory bandwidth optimization still matters for small M where weight matrix is read repeatedly.

## Iteration Results

### Version 1
- **Strategy**: Baseline implementation with unified K_THREADS_SMALL=8 for M <= 8
- **Performance**:
  - M=1: 0.644 TFLOPS
  - M=2-8: 1.376 TFLOPS
  - M=512: 1.125 TFLOPS

### Version 2
- **Strategy**: Reduced K_THREADS_SMALL to 4 for M <= 8 (more N parallelism)
- **Performance**:
  - M=1: 1.296 TFLOPS (101% improvement!)
  - M=2-8: 0.658 TFLOPS (52% regression)
  - M=512: 1.093 TFLOPS

### Version 3 (Final)
- **Strategy**: Hybrid approach combining best configurations
  - **M == 1**: K_THREADS_SMALL=4 (8 N values per warp, max parallelism)
  - **M <= 8**: K_THREADS_SMALL=8 (4 N values per warp, balanced)
  - **M > 8**: TILE_N_LG=256, TILE_M_LG=1 (simple per-thread)

- **Performance**:
  - M=1: **1.296 TFLOPS** (101% vs v1)
  - M=2-8: **1.420 TFLOPS** (3% vs v1)
  - M=512: **1.156 TFLOPS** (3% vs v1)

## Key Optimizations

1. **DP4A intrinsics**: Hardware-accelerated 4-way integer dot product
2. **ILP (Instruction Level Parallelism)**: Process 2 K blocks per call
3. **Warp-level reduction**: Shared partial results across warp threads
4. **Hybrid strategy dispatch**: Different optimal configurations for different M ranges
5. **Template-based kernel**: Compile-time specialization for different K_THREADS_SMALL values

## Correctness
All versions passed correctness with NMSE < 0.05:
- v3 single_token: NMSE=0.000146
- v3 small_batch: NMSE=0.000376
- v3 large_batch: NMSE=0.000204

## Performance Bank

| Config    | Best Version | TFLOPS | Latency (ms) |
|-----------|--------------|----------|---------------|
| M=1       | v3           | 1.296    | 0.811         |
| M=2-8     | v3           | 1.420    | 5.917         |
| M=512     | v3           | 1.156    | 465.279       |

## Conclusion
Version 3 provides the best performance across all M configurations by using a hybrid strategy dispatch that selects the optimal kernel configuration based on runtime M value. The key insight is that different M values benefit from different thread block configurations:
- M=1: Maximizes N dimension parallelism (fewer K threads)
- M=2-8: Balanced approach (moderate K threads)
- M>8: Simplified approach for better occupancy at scale
