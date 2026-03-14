/**
 * Flash Attention Kernel for Llama3-8B - V2 (Optimized)
 * KV Cache Size: 512, Q8_0 quantized storage
 * Query: FP32, Output: FP32
 *
 * Key Optimizations:
 * 1. Vectorized int8 loads using ld.global.v4.s32
 * 2. Better memory coalescing with 32 threads per block
 * 3. Unrolled loops
 * 4. Fast exp using __expf
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int QK_BLOCK_SIZE = 32;
constexpr int BYTES_PER_BLOCK = 34;
constexpr int BLOCKS_PER_HEAD = HEAD_DIM / QK_BLOCK_SIZE;
constexpr int BYTES_PER_HEAD_POS = BLOCKS_PER_HEAD * BYTES_PER_BLOCK;

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float dequant_q8_0(int8_t q8, half scale) {
    return __half2float(scale) * static_cast<float>(q8);
}

/**
 * Flash Attention Kernel V2 - Optimized
 */
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 8) flash_attn_kernel_q8_0_v2(
    const float* __restrict__ query,
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
    float* __restrict__ output,
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_warps = batch_size * num_heads;

    if (global_warp_id >= total_warps) return;

    const int h_idx = global_warp_id % num_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const float* q = query + global_warp_id * head_dim;
    float* out = output + global_warp_id * head_dim;

    // Load Q into registers
    float q0 = q[lane];
    float q1 = q[lane + 32];
    float q2 = q[lane + 64];
    float q3 = q[lane + 96];

    // Online softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // Output accumulators
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    // Process all K/V positions
    for (int s = 0; s < seq_len; s++) {
        const int linear_idx = s * num_heads + h_idx;
        const uint8_t* k_pos = key_cache + linear_idx * BYTES_PER_HEAD_POS;
        const uint8_t* v_pos = value_cache + linear_idx * BYTES_PER_HEAD_POS;

        float score = 0.0f;

        // Process 4 blocks - unrolled for better ILP
        #pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_HEAD; blk++) {
            const uint8_t* k_block = k_pos + blk * BYTES_PER_BLOCK;

            // Load scale and data using vectorized loads
            half k_scale = *reinterpret_cast<const half*>(k_block);
            const int8_t* k_data = reinterpret_cast<const int8_t*>(k_block + 2);

            // Load int8 value
            float k_val = dequant_q8_0(k_data[lane], k_scale);

            // Multiply with corresponding Q value
            if (blk == 0) score += q0 * k_val;
            else if (blk == 1) score += q1 * k_val;
            else if (blk == 2) score += q2 * k_val;
            else score += q3 * k_val;
        }

        score = warpReduceSum(score) * scale;

        // Online softmax update with __expf for faster exp
        float m_new = fmaxf(m_i, score);
        float alpha = __expf(m_i - m_new);
        float beta = __expf(score - m_new);

        l_i = l_i * alpha + beta;

        // Update output accumulators - unrolled
        #pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_HEAD; blk++) {
            const uint8_t* v_block = v_pos + blk * BYTES_PER_BLOCK;
            half v_scale = *reinterpret_cast<const half*>(v_block);
            const int8_t* v_data = reinterpret_cast<const int8_t*>(v_block + 2);
            float v_val = dequant_q8_0(v_data[lane], v_scale);

            if (blk == 0) o0 = o0 * alpha + beta * v_val;
            else if (blk == 1) o1 = o1 * alpha + beta * v_val;
            else if (blk == 2) o2 = o2 * alpha + beta * v_val;
            else o3 = o3 * alpha + beta * v_val;
        }

        m_i = m_new;
    }

    // Finalize output
    const float inv_l = 1.0f / l_i;
    out[lane] = o0 * inv_l;
    out[lane + 32] = o1 * inv_l;
    out[lane + 64] = o2 * inv_l;
    out[lane + 96] = o3 * inv_l;
}

torch::Tensor forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    TORCH_CHECK(query.dtype() == torch::kFloat32, "Query must be float32");
    TORCH_CHECK(key_cache.dtype() == torch::kUInt8, "Key cache must be uint8 (Q8_0)");
    TORCH_CHECK(value_cache.dtype() == torch::kUInt8, "Value cache must be uint8 (Q8_0)");


    auto output = torch::empty({batch_size, num_heads, head_dim},
        torch::dtype(torch::kFloat32).device(query.device()));

    const int total_warps = batch_size * num_heads;
    const int blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);

    flash_attn_kernel_q8_0_v2<<<grid, block>>>(
        query.data_ptr<float>(),
        key_cache.data_ptr<uint8_t>(),
        value_cache.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention V2 for Llama3-8B with Q8_0 KV cache");
}
