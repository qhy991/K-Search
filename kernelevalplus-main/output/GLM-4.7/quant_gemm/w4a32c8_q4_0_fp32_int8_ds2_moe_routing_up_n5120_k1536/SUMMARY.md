# W4A32C8 Q4_0 FP32 INT8 MoE Routing Up Optimization Summary

## Task Overview

**Operator**: Quantized GEMM for DeepSeek-V2 MoE routing expert up/gate projection
**Format**: W4A32C8 with Q4_0×Q8_1 pattern (llama.cpp compatible)
**Dimensions**: N=5120, K=1536, M (variable: 1-512)
**Hardware**: NVIDIA GeForce RTX 4090 (Compute Capability 8.9, 128 SMs)

## Performance Results

### Final Performance (v10 - Best Version)

| Config | M | Performance (TFLOPS) | Latency (ms) | vs Baseline |
|--------|---|---------------------|--------------|-------------|
| single_token | 1   | 1.141               | 0.014        | 46%         |
| small_batch   | 8   | 1.635               | 0.077        | -           |
| large_batch   | 512 | **3.059**           | 2.632        | 1.9%        |

**Baseline** (GGML q4_0 format):
- M=1: 2.47 TFLOPS
- M=512: 162.77 TFLOPS

### Performance Comparison Across Versions

| Version | M=1 TFLOPS | M=8 TFLOPS | M=512 TFLOPS | Key Optimization |
|---------|------------|------------|--------------|-------------------|
| v1      | 0.005      | -          | -            | Initial implementation |
| v2      | 0.005      | 0.025      | 0.464        | Corrected format |
| v3      | 0.005      | 0.017      | 0.503        | Bug fixes |
| v4      | **1.280**  | 0.901      | 1.176        | K-parallelization (small M) |
| v5      | 1.143      | **1.635**  | 2.439        | Dual-path strategy |
| v6      | 1.291      | 0.907      | 1.108        | Attempted optimizations |
| v7      | 1.277      | 0.901      | 1.185        | Tuned block sizes |
| v8      | 1.138      | 1.630      | 1.987        | Tiling attempts |
| v9      | 1.140      | 1.635      | 2.385        | Simplified version |
| **v10** | 1.141      | 1.635      | **3.059**   | **Shared memory tiling** |

## Optimization Journey

### Phase 0: Hardware Profiling
- GPU: RTX 4090, CC 8.9, 128 SMs
- Peak FP32: 82.6 TFLOPS
- Memory Bandwidth: 1008 GB/s
- Ridge Point: 81.9 FLOPs/Byte

**Roofline Analysis** showed compute-bound characteristics for all M values:
- M=1: OI=3.18 FLOPs/Byte (COMPUTE-BOUND)
- M=512: OI=434.20 FLOPs/Byte (COMPUTE-BOUND)

### Phase 1: Understanding BLOCK_Q4_0 Format

**Key discoveries**:
1. BLOCK_Q4_0 is **18 bytes** per block (not 34):
   - 2 bytes: FP16 scale
   - 16 bytes: 32 packed 4-bit values

2. Q4_0 value encoding: [0, 15] represents [-8, +7] (offset by 8)

3. Computation formula (llama.cpp pattern):
   ```
   result = d_w * (d_a * sumi - 8.0 * a_sum)
   ```
   where:
   - `d_w`: weight scale (FP16)
   - `d_a`: activation max/127
   - `sumi`: int8 dot product accumulation
   - `-8.0 * a_sum`: offset compensation

### Phase 2: Initial Implementation Issues

**v1-v3 Problems**:
- Incorrect format understanding (used 34 bytes instead of 18)
- NaN outputs due to wrong unpacking
- Poor memory access patterns

### Phase 3: K-Parallelization Breakthrough (v4)

**Key innovation**: For small M, parallelize across K dimension
- 8 threads per warp work on different K blocks
- Warp shuffle reduction combines partial results
- **Result**: 250x improvement for M=1 (0.005 → 1.28 TFLOPS)

### Phase 4: Dual-Path Strategy (v5)

**Different kernels for different M ranges**:
- M ≤ 16: K-parallelization with warp reduction
- M > 16: Simple 1-thread-per-element with 512 threads/block

**Result**: Achieved 2.44 TFLOPS for M=512

### Phase 5: Shared Memory Tiling (v10 - Final)

**Optimization**: Cache activation blocks in shared memory
- TILE_M_SM=8 rows, TILE_N_SM=64 columns
- Each thread block loads activation into `__shared__ float s_act[TILE_M_SM][QK]`
- Reuses cached data across N dimension

**Result**: 28% improvement for M=512 (2.44 → 3.06 TFLOPS)

## Technical Implementation

### BLOCK_Q4_0 Format
```cpp
// 18 bytes per block of 32 Q4 values
struct BlockQ4_0 {
    uint16_t scale;     // FP16 scale (2 bytes)
    uint8_t  qs[16];   // 32 packed 4-bit values (16 bytes)
};

// Packing: byte[i] = q[i] | (q[i+16] << 4)
// where q[i] in [0, 15] represents actual value in [-8, +7]
```

### Dot Product Computation
```cpp
// 1. Read weight scale
float d_w = read_fp16(w_ptr);

// 2. Compute activation statistics
float a_max = max(|a[0..31]|);
float a_sum = sum(a[0..31]);
float d_a = a_max / 127.0f;

// 3. Quantize activation to INT8
int8_t aq[32];
for (int i = 0; i < 32; i++) {
    aq[i] = clamp(round(a[i] / d_a), -128, 127);
}

// 4. Compute sumi using DP4A
int sumi = 0;
for (int i = 0; i < 4; i++) {
    // Unpack 4 bytes → 8 Q4 values
    int8_t tl[4] = {low nibbles};
    int8_t th[4] = {high nibbles};
    sumi = dp4a(tl, aq[i*4], sumi);
    sumi = dp4a(th, aq[16+i*4], sumi);
}

// 5. Apply llama.cpp formula
return d_w * (d_a * sumi - 8.0f * a_sum);
```

### Kernel Architecture

#### Small M Kernel (M ≤ 16)
```cpp
// Warp-level K-parallelization
// 32 threads per warp:
//   - 4 threads × 8 K-partitions = 32
//   - Each thread processes subset of K blocks
//   - Warp shuffle reduces partial results
```

#### Large M Kernel (M > 16) with Shared Memory
```cpp
// 2D thread block: 64 × 8 threads
// Each thread block:
//   - Loads TILE_M=8 rows of activation (32 values each)
//   - Stores in shared memory: s_act[8][32]
//   - Processes TILE_N=64 output values
//   - Reuses shared memory across K iterations
```

## Key Learnings

1. **Format correctness is critical**: BLOCK_Q4_0 is 18 bytes, not 34
2. **K-parallelization helps small batches**: Reduces latency for M=1 by 250x
3. **Shared memory helps large batches**: 28% improvement for M=512
4. **DP4A instruction is essential**: Efficient int8 dot product
5. **Dual-path strategy optimal**: Different algorithms for different M ranges

## Comparison with Reference Implementation

The att_out experiment achieved 1.85 TFLOPS for M=512 with similar optimizations.
Our kernel achieves 3.02 TFLOPS, which is **63% better** than that reference.

## Baseline Gap Analysis

The baseline shows extremely high performance (162.77 TFLOPS for M=512),
which suggests use of:
- Tensor cores (WMMA)
- Hand-optimized assembly
- Specialized batching techniques

Our implementation at 3.02 TFLOPS represents a practical, maintainable
CUDA kernel using standard optimization techniques.

## Files

- **Best version**: `attempts/w4a32c8_q4_0_fp32_int8_ds2_moe_routing_up_n5120_k1536_v10/kernel.cu`
- **Final version**: `attempts/w4a32c8_q4_0_fp32_int8_ds2_moe_routing_up_n5120_k1536_final/kernel.cu`

## Correctness

All tests pass with NMSE < 0.001:
- single_token: NMSE=0.000261
- small_batch: NMSE=0.000307
- large_batch: NMSE=0.000242
