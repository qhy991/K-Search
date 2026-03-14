# Technical Analysis - W4A32C8 Q4_0 Quantized GEMM

## BLOCK_Q4_0 Format Specification

### Structure (18 bytes per block, 32 values)
```c
struct block_q4_0 {
    uint16_t d;      // FP16 scale (2 bytes)
    uint8_t qs[16];  // Packed 4-bit values (16 bytes)
};
```

### Packing Format
- Each byte stores 2 packed 4-bit values
- `byte[i] = qs[i] (low nibble) | (qs[i+16] << 4) (high nibble)`
- Positions 0-15: stored in low nibbles of bytes 0-15
- Positions 16-31: stored in high nibbles of bytes 0-15

### Unpacking Formula
```c
int q_low = packed_byte & 0x0F;        // position i
int q_high = (packed_byte >> 4) & 0x0F; // position i+16
```

### Dequantization
```
value = d × (q - 8)
```
Where q is in [0, 15], so (q-8) is in [-8, 7].

---

## Kernel Architecture Analysis

### Memory Access Patterns

#### M=1 (Single Token) - Warp-level Strategy
```
Grid: (N/8) blocks
Block: 256 threads = 8 warps
Shared Memory: 5120 floats = 20KB (entire activation)

Algorithm:
1. Cooperatively load activation to shared memory
2. Each warp computes 1 output
3. Lanes in warp process different K blocks (lane-strided)
4. Warp reduction combines partial sums
```

**Performance**: 1.80 TFLOPS (50% of BW-limited max)

#### M≤8 (Small Batch) - Vectorized Strategy
```
Grid: (N/256, M) blocks
Block: 256 threads
Shared Memory: None (or 32 floats per block)

Algorithm:
1. Each thread computes 1 output
2. Vectorized float4 loads for activations
3. Direct global memory reads for weights
```

**Performance**: 1.91 TFLOPS (for M=8)

#### M>8 (Large Batch) - Shared Tiling Strategy
```
Grid: (N/256, M) blocks
Block: 256 threads
Shared Memory: 32 floats per activation block

Algorithm:
1. Load activation block (32 values) to shared memory
2. Each thread computes 1 output
3. Reuse shared memory across all N
```

**Performance**: 2.00 TFLOPS (for M=512)

---

## Performance Optimization Techniques

### 1. Shared Memory Caching (M=1)
```cuda
extern __shared__ float s_activation[];  // K floats

// Cooperatively load
for (int i = tid; i < K; i += blockDim.x) {
    s_activation[i] = activation[i];
}
__syncthreads();
```
**Benefit**: Activation read once, reused by all N outputs

### 2. Warp-level Lane-Strided Loops
```cuda
for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
    // Each lane processes different K block
    // Partial sums accumulated separately
    // Warp reduction at end
}
```
**Benefit**: Better memory latency hiding

### 3. Vectorized Loads
```cuda
float4 a = *reinterpret_cast<const float4*>(&act_block[i * 4]);
a_vals[i * 4 + 0] = a.x;
a_vals[i * 4 + 1] = a.y;
a_vals[i * 4 + 2] = a.z;
a_vals[i * 4 + 3] = a.w;
```
**Benefit**: 128-bit memory transactions

### 4. Warp Reduction
```cuda
#pragma unroll
for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
```
**Benefit**: Efficient partial sum combination

---

## Compute Analysis

### FLOPs Calculation
```
Total FLOPs = 2 × M × N × K
```
For M=1, N=12288, K=5120:
```
FLOPs = 2 × 1 × 12288 × 5120 = 125,829,120
```

### TFLOPS Calculation
```
TFLOPS = FLOPs / (latency_seconds × 10^12)
```
For M=1 at 0.070 ms:
```
TFLOPS = 125,829,120 / (0.000070 × 10^12) = 1.80
```

---

## Bandwidth Analysis

### Data Movement (M=1)
| Component | Size (MB) | Accesses |
|-----------|-----------|----------|
| Weights (Q4_0) | 33.8 | Read once |
| Activation (FP32) | 0.02 | Read once |
| Output (FP32) | 0.05 | Write once |
| **Total** | **33.9** | - |

### Operational Intensity
```
OI = FLOPs / Bytes = 125.8M / 33.9MB = 3.5 FLOPs/Byte
```

### BW-Limited Max Performance
```
Max TFLOPS = OI × Peak_BW = 3.5 × 1008 GB/s / 1000 = 3.6 TFLOPS
```

**Achieved**: 1.80 TFLOPS = **50% of BW-limited max**

---

## Correctness Verification

### Reference Implementation
The test uses `reference_q4_0_fp32_gemm()` from `llm_kernel_test/reference/gemm_ref.py`:

```python
# Dequantize weights
w_scales = extract_fp16_scale(weight_blocks)
w_qs = unpack_q4_0_values(weight_blocks)
W_dequant = w_scales × (w_qs - 8)

# FP32 matrix multiplication
output = activation @ W_dequant.T
```

### NMSE Calculation
```
NMSE = ||output_ref - output_kernel||² / ||output_ref||²
```

**Threshold**: NMSE < 0.05

### Our Results
All versions pass with **NMSE = 0.000000** (essentially perfect match).

---

## Performance Gap Analysis

### vs Baseline (RTX 4090)
| M | Our TFLOPS | Baseline TFLOPS | Gap |
|---|-------------|-----------------|-----|
| 1 | 1.80 | 8.54 | 5x slower |
| 512 | 2.00 | 234.7 | 117x slower |

### Possible Reasons for Gap

1. **DP4A Instruction**: Baseline likely uses INT8 dot products
   - Q4_0 values unpacked to INT8
   - 4 DP4A instructions per 32 values
   - Significantly faster than scalar approach

2. **Tensor Cores**: For large M, baseline may use:
   - WMMA (Warp Matrix Multiply-Accumulate)
   - Matrix co-processor units
   - Much higher throughput

3. **Memory Tiling**: More sophisticated:
   - 2D tiles across N and K
   - Weight tiles in shared memory
   - Better cache utilization

4. **Activation Quantization**: Baseline may:
   - Dynamically quantize activation to Q8_1
   - Use Q8_1 × Q4_0 optimized routine
   - Better compute efficiency

---

## Code Quality Metrics

| Metric | Value |
|--------|-------|
| Lines of Code (v8) | 217 |
| Kernels | 3 (adaptive) |
| Correctness | ✅ PASS |
| Max Performance | 1.80 TFLOPS (M=1) |
| Code Clarity | High |
| Maintainability | High |

---

## Future Work

### Near-term (Easy Wins)
1. **Add DP4A path**: Unpack Q4_0 to INT8, use DP4A
2. **Better grid calculation**: Optimize blocks/SM ratio
3. **Prefetch weights**: Overlap compute with memory loads

### Medium-term (Significant Gains)
1. **2D Tiling**: Tile both N and K dimensions
2. **Weight tiles in shared memory**: Cache frequently accessed weights
3. **Persistent threads**: Better data reuse

### Long-term (Advanced)
1. **Tensor Core usage**: WMMA for INT8 compute
2. **Activation quantization**: Q8_1-style dynamic quantization
3. **Multi-kernel fusion**: Fuse dequantization with GEMM

---

## References

1. **llama.cpp**: `vec_dot_q4_0_q8_1` implementation
2. **CUDA Programming Guide**: Warp shuffle, shared memory
3. **Roofline Model**: Performance analysis methodology
4. **RTX 4090 Specification**: 82.6 TFLOPS FP32, 1008 GB/s BW
