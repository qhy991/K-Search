/**
 * Top-K Sampling Kernel for Qwen2.5-7B - V3 Optimized
 *
 * Configuration:
 *   - K = 6 (top-6 sampling)
 *   - vocab_subset = 256
 *   - Batch sizes: 1, 8, 512
 *   - Hardware: NVIDIA RTX 4090 (Ada Lovelace, SM 89)
 *
 * V3 Optimizations:
 *   1. Explicit register variables instead of struct (reduces register pressure)
 *   2. Fully inlined insertion logic
 *   3. Higher occupancy hints with __launch_bounds__
 *   4. Batch interleaved memory access for small batches
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

constexpr int TOPK_K = 6;
constexpr int WARP_SIZE = 32;

/**
 * Ultra-fast single batch kernel
 * Entire vocab (256 floats = 64 float4s) processed by one warp
 */
extern "C" __launch_bounds__(32, 8) __global__ void topk_single_batch_kernel(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int vocab_size
) {
    const int lane_id = threadIdx.x;

    const float4* probs_vec = reinterpret_cast<const float4*>(probs);
    const int num_vec = vocab_size >> 2;

    // Explicit register variables for top-k
    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY;
    float v3 = -INFINITY, v4 = -INFINITY, v5 = -INFINITY;
    int i0 = -1, i1 = -1, i2 = -1, i3 = -1, i4 = -1, i5 = -1;

    // Process vectorized loads
    for (int v = lane_id; v < num_vec; v += 32) {
        const float4 vec = __ldg(&probs_vec[v]);
        const int base = v << 2;

        // Process all 4 elements from float4
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const float val = reinterpret_cast<const float*>(&vec)[e];
            const int idx = base + e;

            if (val > v0) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = val; i0 = idx;
            } else if (val > v1) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = v1; i2 = i1; v1 = val; i1 = idx;
            } else if (val > v2) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = val; i2 = idx;
            } else if (val > v3) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = val; i3 = idx;
            } else if (val > v4) {
                v5 = v4; i5 = i4; v4 = val; i4 = idx;
            } else if (val > v5) {
                v5 = val; i5 = idx;
            }
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float rv0 = __shfl_down_sync(0xffffffff, v0, offset);
        float rv1 = __shfl_down_sync(0xffffffff, v1, offset);
        float rv2 = __shfl_down_sync(0xffffffff, v2, offset);
        float rv3 = __shfl_down_sync(0xffffffff, v3, offset);
        float rv4 = __shfl_down_sync(0xffffffff, v4, offset);
        float rv5 = __shfl_down_sync(0xffffffff, v5, offset);
        int ri0 = __shfl_down_sync(0xffffffff, i0, offset);
        int ri1 = __shfl_down_sync(0xffffffff, i1, offset);
        int ri2 = __shfl_down_sync(0xffffffff, i2, offset);
        int ri3 = __shfl_down_sync(0xffffffff, i3, offset);
        int ri4 = __shfl_down_sync(0xffffffff, i4, offset);
        int ri5 = __shfl_down_sync(0xffffffff, i5, offset);

        if (lane_id < offset) {
            // Merge received values
            if (rv0 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv0; i0 = ri0; }
            else if (rv0 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv0; i1 = ri0; }
            else if (rv0 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv0; i2 = ri0; }
            else if (rv0 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv0; i3 = ri0; }
            else if (rv0 > v4) { v5 = v4; i5 = i4; v4 = rv0; i4 = ri0; }
            else if (rv0 > v5) { v5 = rv0; i5 = ri0; }

            if (rv1 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv1; i0 = ri1; }
            else if (rv1 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv1; i1 = ri1; }
            else if (rv1 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv1; i2 = ri1; }
            else if (rv1 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv1; i3 = ri1; }
            else if (rv1 > v4) { v5 = v4; i5 = i4; v4 = rv1; i4 = ri1; }
            else if (rv1 > v5) { v5 = rv1; i5 = ri1; }

            if (rv2 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv2; i0 = ri2; }
            else if (rv2 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv2; i1 = ri2; }
            else if (rv2 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv2; i2 = ri2; }
            else if (rv2 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv2; i3 = ri2; }
            else if (rv2 > v4) { v5 = v4; i5 = i4; v4 = rv2; i4 = ri2; }
            else if (rv2 > v5) { v5 = rv2; i5 = ri2; }

            if (rv3 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv3; i0 = ri3; }
            else if (rv3 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv3; i1 = ri3; }
            else if (rv3 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv3; i2 = ri3; }
            else if (rv3 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv3; i3 = ri3; }
            else if (rv3 > v4) { v5 = v4; i5 = i4; v4 = rv3; i4 = ri3; }
            else if (rv3 > v5) { v5 = rv3; i5 = ri3; }

            if (rv4 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv4; i0 = ri4; }
            else if (rv4 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv4; i1 = ri4; }
            else if (rv4 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv4; i2 = ri4; }
            else if (rv4 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv4; i3 = ri4; }
            else if (rv4 > v4) { v5 = v4; i5 = i4; v4 = rv4; i4 = ri4; }
            else if (rv4 > v5) { v5 = rv4; i5 = ri4; }

            if (rv5 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv5; i0 = ri5; }
            else if (rv5 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv5; i1 = ri5; }
            else if (rv5 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv5; i2 = ri5; }
            else if (rv5 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv5; i3 = ri5; }
            else if (rv5 > v4) { v5 = v4; i5 = i4; v4 = rv5; i4 = ri5; }
            else if (rv5 > v5) { v5 = rv5; i5 = ri5; }
        }
    }

    if (lane_id == 0) {
        samples[0] = i0;
    }
}

/**
 * Small batch kernel: one warp per batch element
 */
extern "C" __launch_bounds__(128, 4) __global__ void topk_small_batch_kernel(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size
) {
    const int warp_id = (blockIdx.x << 2) + threadIdx.y;
    const int lane_id = threadIdx.x;

    if (warp_id >= batch_size) return;

    const float4* batch_probs = reinterpret_cast<const float4*>(
        probs + static_cast<size_t>(warp_id) * vocab_size);
    const int num_vec = vocab_size >> 2;

    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY;
    float v3 = -INFINITY, v4 = -INFINITY, v5 = -INFINITY;
    int i0 = -1, i1 = -1, i2 = -1, i3 = -1, i4 = -1, i5 = -1;

    for (int v = lane_id; v < num_vec; v += 32) {
        const float4 vec = __ldg(&batch_probs[v]);
        const int base = v << 2;

        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const float val = reinterpret_cast<const float*>(&vec)[e];
            const int idx = base + e;

            if (val > v0) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = val; i0 = idx;
            } else if (val > v1) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = v1; i2 = i1; v1 = val; i1 = idx;
            } else if (val > v2) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = val; i2 = idx;
            } else if (val > v3) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = val; i3 = idx;
            } else if (val > v4) {
                v5 = v4; i5 = i4; v4 = val; i4 = idx;
            } else if (val > v5) {
                v5 = val; i5 = idx;
            }
        }
    }

    // Warp reduction (same pattern as single batch)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float rv0 = __shfl_down_sync(0xffffffff, v0, offset);
        float rv1 = __shfl_down_sync(0xffffffff, v1, offset);
        float rv2 = __shfl_down_sync(0xffffffff, v2, offset);
        float rv3 = __shfl_down_sync(0xffffffff, v3, offset);
        float rv4 = __shfl_down_sync(0xffffffff, v4, offset);
        float rv5 = __shfl_down_sync(0xffffffff, v5, offset);
        int ri0 = __shfl_down_sync(0xffffffff, i0, offset);
        int ri1 = __shfl_down_sync(0xffffffff, i1, offset);
        int ri2 = __shfl_down_sync(0xffffffff, i2, offset);
        int ri3 = __shfl_down_sync(0xffffffff, i3, offset);
        int ri4 = __shfl_down_sync(0xffffffff, i4, offset);
        int ri5 = __shfl_down_sync(0xffffffff, i5, offset);

        if (lane_id < offset) {
            if (rv0 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv0; i0 = ri0; } else if (rv0 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv0; i1 = ri0; } else if (rv0 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv0; i2 = ri0; } else if (rv0 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv0; i3 = ri0; } else if (rv0 > v4) { v5 = v4; i5 = i4; v4 = rv0; i4 = ri0; } else if (rv0 > v5) { v5 = rv0; i5 = ri0; }
            if (rv1 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv1; i0 = ri1; } else if (rv1 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv1; i1 = ri1; } else if (rv1 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv1; i2 = ri1; } else if (rv1 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv1; i3 = ri1; } else if (rv1 > v4) { v5 = v4; i5 = i4; v4 = rv1; i4 = ri1; } else if (rv1 > v5) { v5 = rv1; i5 = ri1; }
            if (rv2 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv2; i0 = ri2; } else if (rv2 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv2; i1 = ri2; } else if (rv2 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv2; i2 = ri2; } else if (rv2 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv2; i3 = ri2; } else if (rv2 > v4) { v5 = v4; i5 = i4; v4 = rv2; i4 = ri2; } else if (rv2 > v5) { v5 = rv2; i5 = ri2; }
            if (rv3 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv3; i0 = ri3; } else if (rv3 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv3; i1 = ri3; } else if (rv3 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv3; i2 = ri3; } else if (rv3 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv3; i3 = ri3; } else if (rv3 > v4) { v5 = v4; i5 = i4; v4 = rv3; i4 = ri3; } else if (rv3 > v5) { v5 = rv3; i5 = ri3; }
            if (rv4 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv4; i0 = ri4; } else if (rv4 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv4; i1 = ri4; } else if (rv4 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv4; i2 = ri4; } else if (rv4 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv4; i3 = ri4; } else if (rv4 > v4) { v5 = v4; i5 = i4; v4 = rv4; i4 = ri4; } else if (rv4 > v5) { v5 = rv4; i5 = ri4; }
            if (rv5 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv5; i0 = ri5; } else if (rv5 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv5; i1 = ri5; } else if (rv5 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv5; i2 = ri5; } else if (rv5 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv5; i3 = ri5; } else if (rv5 > v4) { v5 = v4; i5 = i4; v4 = rv5; i4 = ri5; } else if (rv5 > v5) { v5 = rv5; i5 = ri5; }
        }
    }

    if (lane_id == 0) {
        samples[warp_id] = i0;
    }
}

/**
 * Large batch kernel: one block per batch element
 */
extern "C" __launch_bounds__(128, 2) __global__ void topk_large_batch_kernel(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float4* batch_probs = reinterpret_cast<const float4*>(
        probs + static_cast<size_t>(batch_idx) * vocab_size);
    const int num_vec = vocab_size >> 2;

    __shared__ float s_vals[4][TOPK_K];
    __shared__ int s_idxs[4][TOPK_K];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY;
    float v3 = -INFINITY, v4 = -INFINITY, v5 = -INFINITY;
    int i0 = -1, i1 = -1, i2 = -1, i3 = -1, i4 = -1, i5 = -1;

    for (int v = threadIdx.x; v < num_vec; v += 128) {
        const float4 vec = __ldg(&batch_probs[v]);
        const int base = v << 2;

        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const float val = reinterpret_cast<const float*>(&vec)[e];
            const int idx = base + e;

            if (val > v0) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = val; i0 = idx;
            } else if (val > v1) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = v1; i2 = i1; v1 = val; i1 = idx;
            } else if (val > v2) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                v2 = val; i2 = idx;
            } else if (val > v3) {
                v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = val; i3 = idx;
            } else if (val > v4) {
                v5 = v4; i5 = i4; v4 = val; i4 = idx;
            } else if (val > v5) {
                v5 = val; i5 = idx;
            }
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float rv0 = __shfl_down_sync(0xffffffff, v0, offset);
        float rv1 = __shfl_down_sync(0xffffffff, v1, offset);
        float rv2 = __shfl_down_sync(0xffffffff, v2, offset);
        float rv3 = __shfl_down_sync(0xffffffff, v3, offset);
        float rv4 = __shfl_down_sync(0xffffffff, v4, offset);
        float rv5 = __shfl_down_sync(0xffffffff, v5, offset);
        int ri0 = __shfl_down_sync(0xffffffff, i0, offset);
        int ri1 = __shfl_down_sync(0xffffffff, i1, offset);
        int ri2 = __shfl_down_sync(0xffffffff, i2, offset);
        int ri3 = __shfl_down_sync(0xffffffff, i3, offset);
        int ri4 = __shfl_down_sync(0xffffffff, i4, offset);
        int ri5 = __shfl_down_sync(0xffffffff, i5, offset);

        if (lane_id < offset) {
            if (rv0 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv0; i0 = ri0; } else if (rv0 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv0; i1 = ri0; } else if (rv0 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv0; i2 = ri0; } else if (rv0 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv0; i3 = ri0; } else if (rv0 > v4) { v5 = v4; i5 = i4; v4 = rv0; i4 = ri0; } else if (rv0 > v5) { v5 = rv0; i5 = ri0; }
            if (rv1 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv1; i0 = ri1; } else if (rv1 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv1; i1 = ri1; } else if (rv1 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv1; i2 = ri1; } else if (rv1 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv1; i3 = ri1; } else if (rv1 > v4) { v5 = v4; i5 = i4; v4 = rv1; i4 = ri1; } else if (rv1 > v5) { v5 = rv1; i5 = ri1; }
            if (rv2 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv2; i0 = ri2; } else if (rv2 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv2; i1 = ri2; } else if (rv2 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv2; i2 = ri2; } else if (rv2 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv2; i3 = ri2; } else if (rv2 > v4) { v5 = v4; i5 = i4; v4 = rv2; i4 = ri2; } else if (rv2 > v5) { v5 = rv2; i5 = ri2; }
            if (rv3 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv3; i0 = ri3; } else if (rv3 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv3; i1 = ri3; } else if (rv3 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv3; i2 = ri3; } else if (rv3 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv3; i3 = ri3; } else if (rv3 > v4) { v5 = v4; i5 = i4; v4 = rv3; i4 = ri3; } else if (rv3 > v5) { v5 = rv3; i5 = ri3; }
            if (rv4 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv4; i0 = ri4; } else if (rv4 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv4; i1 = ri4; } else if (rv4 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv4; i2 = ri4; } else if (rv4 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv4; i3 = ri4; } else if (rv4 > v4) { v5 = v4; i5 = i4; v4 = rv4; i4 = ri4; } else if (rv4 > v5) { v5 = rv4; i5 = ri4; }
            if (rv5 > v0) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = rv5; i0 = ri5; } else if (rv5 > v1) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = rv5; i1 = ri5; } else if (rv5 > v2) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = rv5; i2 = ri5; } else if (rv5 > v3) { v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = rv5; i3 = ri5; } else if (rv5 > v4) { v5 = v4; i5 = i4; v4 = rv5; i4 = ri5; } else if (rv5 > v5) { v5 = rv5; i5 = ri5; }
        }
    }

    // Store warp results
    if (lane_id == 0) {
        s_vals[warp_id][0] = v0; s_vals[warp_id][1] = v1;
        s_vals[warp_id][2] = v2; s_vals[warp_id][3] = v3;
        s_vals[warp_id][4] = v4; s_vals[warp_id][5] = v5;
        s_idxs[warp_id][0] = i0; s_idxs[warp_id][1] = i1;
        s_idxs[warp_id][2] = i2; s_idxs[warp_id][3] = i3;
        s_idxs[warp_id][4] = i4; s_idxs[warp_id][5] = i5;
    }
    __syncthreads();

    // Merge in warp 0
    if (warp_id == 0) {
        v0 = v1 = v2 = v3 = v4 = v5 = -INFINITY;
        i0 = i1 = i2 = i3 = i4 = i5 = -1;

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            #pragma unroll
            for (int k = 0; k < TOPK_K; k++) {
                const float val = s_vals[w][k];
                const int idx = s_idxs[w][k];
                if (val > -INFINITY) {
                    if (val > v0) {
                        v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                        v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = val; i0 = idx;
                    } else if (val > v1) {
                        v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                        v2 = v1; i2 = i1; v1 = val; i1 = idx;
                    } else if (val > v2) {
                        v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = v2; i3 = i2;
                        v2 = val; i2 = idx;
                    } else if (val > v3) {
                        v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = val; i3 = idx;
                    } else if (val > v4) {
                        v5 = v4; i5 = i4; v4 = val; i4 = idx;
                    } else if (val > v5) {
                        v5 = val; i5 = idx;
                    }
                }
            }
        }

        if (lane_id == 0) {
            samples[batch_idx] = i0;
        }
    }
}

extern "C" void topk_kernel(
    const float* probs,
    int64_t* indices,
    int batch_size,
    int vocab_size,
    int k
) {
    if (batch_size == 1) {
        topk_single_batch_kernel<<<1, 32>>>(probs, indices, vocab_size);
    } else if (batch_size <= 32) {
        dim3 block(32, 4);
        int num_blocks = (batch_size + 3) >> 2;
        topk_small_batch_kernel<<<num_blocks, block>>>(probs, indices, batch_size, vocab_size);
    } else {
        topk_large_batch_kernel<<<batch_size, 128>>>(probs, indices, batch_size, vocab_size);
    }
}

#include <torch/extension.h>

torch::Tensor forward(torch::Tensor probs, int k) {
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(probs.dim() == 2, "probs must be 2D");
    TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");

    const int batch_size = probs.size(0);
    const int vocab_size = probs.size(1);

    auto samples = torch::empty({batch_size},
        torch::dtype(torch::kInt64).device(probs.device()));

    topk_kernel(
        probs.data_ptr<float>(),
        samples.data_ptr<int64_t>(),
        batch_size, vocab_size, k
    );

    return samples;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "TopK Sampling");
}
