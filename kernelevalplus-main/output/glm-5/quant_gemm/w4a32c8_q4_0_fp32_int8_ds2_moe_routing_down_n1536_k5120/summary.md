# W4A32C8 Q4_0 × FP32 Quantized GEMM Summary

## Task Definition
- **Name**: w4a32c8_q4_0_fp32_int8_ds2_moe_routing_down_n1536_k5120
- **Operator**: Quantized GEMM (W4A32C8 variant)
- **Architecture**: DeepSeek-V2 MoE Routing Down Projection
- **Dimensions**: N=1536, K=5120, M=variable

## Q4_0 Format (18 bytes per block)
- **scale**: FP16 (2 bytes, offset 0)
- **qs**: packed 4-bit values (16 bytes, offset 2)
- **Block size**: 32 elements
- **Encoding**: q = round(val / scale + 8), q ∈ [0, 15]
- **Decoding**: val = scale × (q - 8), val ∈ [-8×scale, 7×scale]

## Reference Implementation
The test framework uses `reference_q4_0_fp32_gemm` which:
1. Dequantizes Q4_0 weights to FP32
2. Performs full FP32 matrix multiplication

Note: Despite the task definition mentioning "llama.cpp Q4_0×Q8_1 pattern", the actual reference used is full FP32 dequantization, not dynamic Q8_1 quantization.

## Performance Results

| Version | M=1 (TFLOPS) | M=512 (TFLOPS) | NMSE | Notes |
|---------|----------------|-----------------|------|-------|
| v1 (small+large) | 0.078 | 1.913 | 0.000 | Basic implementation |
| v2 (llama_cpp Q8_1) | 0.081 | 1.898 | 0.0006 | Different formula, slight numerical difference |
| v3 (optimized) | 0.15 | 1.834 | 0.000 | Aggressive unrolling, precomputed dequantization |
| v4 (memory opt) | 0.15 | 1.847 | 0.000 | Memory access optimizations |
| **Final** | **0.15** | **1.805** | **0.000** | Combined with strategy dispatch |

## Baseline Comparison
- **M=1**: Baseline 2.48 TFLOPS vs Kernel 0.15 TFLOPS (6% of baseline)
- **M=512**: Baseline 155.48 TFLOPS vs Kernel 1.805 TFLOPS (1.2% of baseline)

Note: The baseline for M=512 (155.48 TFLOPS) seems unusually high for Q4_0 quantized GEMM, suggesting it may be using a different algorithm or library (cuBLAS).

## Roofline Analysis (RTX 4090)
- **Peak FP32**: ~82.6 TFLOPS
- **Peak Bandwidth**: ~1008 GB/s
- **Ridge Point**: ~82 FLOPs/Byte

**For M=1**:
- Operational Intensity: 3.54 FLOPs/Byte
- Regime: Memory-bound (OI << Ridge)
- Theoretical max (BW-limited): 1008 × 3.54 ≈ 3.57 TFLOPS
- Achievement: 0.15 TFLOPS = 4.2% of theoretical

**For M=512**:
- Operational Intensity: 540 FLOPs/Byte
- Regime: Compute-bound (OI >> Ridge)
- Theoretical max (compute-limited): 82.6 TFLOPS
- Achievement: 1.805 TFLOPS = 2.2% of theoretical

## Kernel Implementation Details

### Strategy Dispatch
- **Small M (M <= 8)**: Memory-bound kernel with direct memory access
- **Large M (M > 8)**: Compute-bound kernel with same optimizations

### Key Optimizations
1. **Precomputed dequantization**: Weight scale applied once per block
2. **Aggressive loop unrolling**: Inner 16 iterations fully unrolled
3. **Efficient memory access**: Coalesced global memory reads
4. **Strategy dispatch**: Different kernels for different batch sizes

### Limitations
1. **Memory-bound for small M**: Single-token performance limited by Q4_0 unpacking overhead
2. **No DP4A usage**: Requires INT8 activation (Q8_1 format)
3. **No tensor cores**: Using standard FP32 compute

## Final Kernel Location
`/home/qinhaiyan/kernelevalplus/output/glm-5/quant_gemm/w4a32c8_q4_0_fp32_int8_ds2_moe_routing_down_n1536_k5120/kernel.cu`

## Correctness Status
✅ **PASS** - All test configurations pass with NMSE = 0.000
- single_token: ✅ NMSE=0.000
- small_batch: ✅ NMSE=0.000
- large_batch: ✅ NMSE=0.000
