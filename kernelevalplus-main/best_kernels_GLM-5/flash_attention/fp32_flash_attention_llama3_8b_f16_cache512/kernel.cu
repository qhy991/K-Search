/**
 * Flash Attention Kernel for Llama3-8B - Optimized Final Version
 * KV Cache Size: 512, F16 storage
 * Query: FP32, Output: FP32
 *
 * Performance: 17.0% of GGML baseline (RTX 4090)
 * - batch_1: 0.185 TFLOPS (baseline: 1.09)
 * - batch_8: 1.416 TFLOPS (baseline: 3.68)
 * - batch_512: 4.702 TFLOPS (baseline: 130.41)
 *
 * Key Optimizations:
 * 1. #pragma unroll 4 for loop optimization (key performance improvement)
 * 2. 4 warps per block for higher occupancy
 * 3. __expf for faster exponential computation
 * 4. Online softmax for single-pass computation
 * 5. Strided Q access pattern
 * 6. Minimal register pressure
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 8) flash_attn_kernel(
    const float* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
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
    const half* k_base = key_cache + h_idx * head_dim;
    const half* v_base = value_cache + h_idx * head_dim;
    float* out = output + global_warp_id * head_dim;

    const int head_stride = num_heads * head_dim;

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

    // Process all K/V positions with unrolling for ILP
    #pragma unroll 4
    for (int s = 0; s < seq_len; s++) {
        const half* k = k_base + s * head_stride;
        const half* v = v_base + s * head_stride;

        // Compute Q @ K[s]^T
        float score = q0 * __half2float(k[lane])
                    + q1 * __half2float(k[lane + 32])
                    + q2 * __half2float(k[lane + 64])
                    + q3 * __half2float(k[lane + 96]);
        score = warpReduceSum(score) * scale;

        // Online softmax update
        float m_new = fmaxf(m_i, score);
        float alpha = __expf(m_i - m_new);
        float beta = __expf(score - m_new);

        l_i = l_i * alpha + beta;

        o0 = o0 * alpha + beta * __half2float(v[lane]);
        o1 = o1 * alpha + beta * __half2float(v[lane + 32]);
        o2 = o2 * alpha + beta * __half2float(v[lane + 64]);
        o3 = o3 * alpha + beta * __half2float(v[lane + 96]);

        m_i = m_new;
    }

    // Write output
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
    TORCH_CHECK(key_cache.dtype() == torch::kFloat16, "Key cache must be float16");
    TORCH_CHECK(value_cache.dtype() == torch::kFloat16, "Value cache must be float16");

    auto output = torch::empty({batch_size, num_heads, head_dim},
        torch::dtype(torch::kFloat32).device(query.device()));

    const int total_warps = batch_size * num_heads;
    const int blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);

    flash_attn_kernel<<<grid, block>>>(
        query.data_ptr<float>(),
        reinterpret_cast<const half*>(key_cache.data_ptr()),
        reinterpret_cast<const half*>(value_cache.data_ptr()),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention for Llama3-8B with F16 KV cache");
}
