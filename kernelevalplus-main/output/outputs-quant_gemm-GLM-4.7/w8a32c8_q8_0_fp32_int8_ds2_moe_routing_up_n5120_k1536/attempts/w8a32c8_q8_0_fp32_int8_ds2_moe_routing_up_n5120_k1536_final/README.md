# Quantized GEMM Kernel: w8a32c8_q8_0_fp32_int8_ds2_moe_routing_up_n5120_k1536

## Task Summary

**Operator**: Quantized GEMM with Q8_0 block quantization  
**Model**: DeepSeek V2 MoE Routing Up Projection  
**Dimensions**: N=5120, K=1536, M=variable (batch)  
**Format**: W8A32C8 (8-bit weights, 32-bit activation, 32-bit compute)

## Performance Results

| M (batch) | Latency (ms) | TFLOPS | Improvement |
|-----------|--------------|--------|-------------|
| 1         | 0.028        | 0.57   | +6.5%       |
| 8         | 0.065        | 1.935  | +0.1%       |
| 512       | 2.873        | 2.803  | +2.4%       |

## Optimization Techniques

### 1. Roofline Analysis
- **Operational Intensity (M=1)**: 1.88 FLOPs/Byte
- **Ridge Point (FP32)**: 81.9 FLOPs/Byte
- **Conclusion**: Memory-bound for M ≤ 32, compute-bound for M ≥ 64

### 2. Dynamic Kernel Dispatch
```cuda
if (M <= 32) {
    // Memory-bound: Use unrolled kernel
    q8_0_gemm_unrolled<1><<<grid, block>>>(...);
} else {
    // Compute-bound: Use standard kernel
    q8_0_gemm_standard<1><<<grid, block>>>(...);
}
```

### 3. K-Loop Unrolling (Memory-Bound Cases)
- Process 2 K blocks per iteration
- Reduces loop overhead and improves ILP
- Better memory bandwidth utilization

### 4. Optimal Thread Configuration
- Thread block: 16×16 = 256 threads
- N_PER_THREAD: 1 (one output per thread)
- Grid: Calculated based on M and N

## Q8_0 Format

Each block contains 32 int8 values with a shared FP16 scale:
- Total size: 34 bytes per block
- Scale: 2 bytes (FP16)
- Quantized values: 32 bytes (int8[32])

Computation: `output = sum(activation[i] * weight_qs[i]) * scale`

## Development History

| Version | Key Changes | M=1 TFLOPS | M=512 TFLOPS |
|---------|-------------|------------|--------------|
| v1      | Basic 2D kernel | 0.535 | 2.788 |
| v2-v6   | Various optimizations | - | - |
| v7      | Shared memory (failed correctness) | - | - |
| v9      | K-loop unrolling | 0.57 | 2.703 |
| v10     | Combined dispatch | 0.57 | 2.765 |
| **final** | **Optimized combined** | **0.57** | **2.803** |

## Testing

All test configurations pass correctness (NMSE < 0.05):
- Single token (M=1)
- Small batch (M=8)
- Large batch (M=512)

## Files

- `kernel.cu` - Main CUDA kernel implementation
- `test_results.json` - Test results with performance metrics
