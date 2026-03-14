/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel
 * v11 - Advanced optimizations with vectorized loads and prefetching
 *
 * Optimizations:
 * 1. Vectorized float4 loads with __ldg() for texture cache
 * 2. Software pipelining with weight prefetching
 * 3. Register blocking for better ILP
 * 4. Optimized for Ada architecture (RTX 4090)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Prefetch helper
__device__ __forceinline__ void prefetch_block(const block_q4_0* ptr) {
    // Prefetch 18 bytes (Q4_0 block size) using 32-byte cache line
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}

// ============================================================================
// Optimized kernel with prefetching and vectorized loads
// ============================================================================
__global__ void __launch_bounds__(64) gemm_q4_0_fp32_prefetch(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const block_q4_0* __restrict__ w_row = (const block_q4_0* __restrict__)weight_q + n * num_blocks_k;
    const float4* __restrict__ act_row_vec = (const float4* __restrict__)(activation + m * K);

    float sum = 0.0f;

    // Prefetch first weight block
    prefetch_block(&w_row[0]);

    // Process all K blocks with software pipelining
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Prefetch next block
        if (kb + 1 < num_blocks_k) {
            prefetch_block(&w_row[kb + 1]);
        }

        // Load current weight block
        const block_q4_0 w_block = w_row[kb];
        const float scale = half_to_float(w_block.d);

        // Vectorized load of activations using texture cache
        const float4* act_vec = act_row_vec + kb * 8;  // 8 float4 = 32 floats

        // Load 8 float4 vectors with texture cache hint
        float4 v0 = __ldg(&act_vec[0]);
        float4 v1 = __ldg(&act_vec[1]);
        float4 v2 = __ldg(&act_vec[2]);
        float4 v3 = __ldg(&act_vec[3]);
        float4 v4 = __ldg(&act_vec[4]);
        float4 v5 = __ldg(&act_vec[5]);
        float4 v6 = __ldg(&act_vec[6]);
        float4 v7 = __ldg(&act_vec[7]);

        // Compute using the unpacked values
        // Map qs indices to vector components
        // qs[0]: low -> v0.x (idx 0), high -> v4.y (idx 17)
        // qs[1]: low -> v0.y (idx 1), high -> v4.z (idx 18)
        // etc.
        
        float b0 = (float)((int8_t)(w_block.qs[0] & 0x0F) - 8) * v0.x
                 + (float)((int8_t)(w_block.qs[0] >> 4) - 8) * v4.y;
        float b1 = (float)((int8_t)(w_block.qs[1] & 0x0F) - 8) * v0.y
                 + (float)((int8_t)(w_block.qs[1] >> 4) - 8) * v4.z;
        float b2 = (float)((int8_t)(w_block.qs[2] & 0x0F) - 8) * v0.z
                 + (float)((int8_t)(w_block.qs[2] >> 4) - 8) * v4.w;
        float b3 = (float)((int8_t)(w_block.qs[3] & 0x0F) - 8) * v0.w
                 + (float)((int8_t)(w_block.qs[3] >> 4) - 8) * v5.x;
        float b4 = (float)((int8_t)(w_block.qs[4] & 0x0F) - 8) * v1.x
                 + (float)((int8_t)(w_block.qs[4] >> 4) - 8) * v5.y;
        float b5 = (float)((int8_t)(w_block.qs[5] & 0x0F) - 8) * v1.y
                 + (float)((int8_t)(w_block.qs[5] >> 4) - 8) * v5.z;
        float b6 = (float)((int8_t)(w_block.qs[6] & 0x0F) - 8) * v1.z
                 + (float)((int8_t)(w_block.qs[6] >> 4) - 8) * v5.w;
        float b7 = (float)((int8_t)(w_block.qs[7] & 0x0F) - 8) * v1.w
                 + (float)((int8_t)(w_block.qs[7] >> 4) - 8) * v6.x;
        float b8 = (float)((int8_t)(w_block.qs[8] & 0x0F) - 8) * v2.x
                 + (float)((int8_t)(w_block.qs[8] >> 4) - 8) * v6.y;
        float b9 = (float)((int8_t)(w_block.qs[9] & 0x0F) - 8) * v2.y
                 + (float)((int8_t)(w_block.qs[9] >> 4) - 8) * v6.z;
        float b10 = (float)((int8_t)(w_block.qs[10] & 0x0F) - 8) * v2.z
                  + (float)((int8_t)(w_block.qs[10] >> 4) - 8) * v6.w;
        float b11 = (float)((int8_t)(w_block.qs[11] & 0x0F) - 8) * v2.w
                  + (float)((int8_t)(w_block.qs[11] >> 4) - 8) * v7.x;
        float b12 = (float)((int8_t)(w_block.qs[12] & 0x0F) - 8) * v3.x
                  + (float)((int8_t)(w_block.qs[12] >> 4) - 8) * v7.y;
        float b13 = (float)((int8_t)(w_block.qs[13] & 0x0F) - 8) * v3.y
                  + (float)((int8_t)(w_block.qs[13] >> 4) - 8) * v7.z;
        float b14 = (float)((int8_t)(w_block.qs[14] & 0x0F) - 8) * v3.z
                  + (float)((int8_t)(w_block.qs[14] >> 4) - 8) * v7.w;
        float b15 = (float)((int8_t)(w_block.qs[15] & 0x0F) - 8) * v3.w
                  + (float)((int8_t)(w_block.qs[15] >> 4) - 8) * v4.x;

        sum += scale * (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 +
                        b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15);
    }

    output[m * N + n] = sum;
}

// Large batch kernel with vectorized loads
__global__ void __launch_bounds__(256) gemm_q4_0_fp32_large_batch_v11(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const block_q4_0* __restrict__ w_row = (const block_q4_0* __restrict__)weight_q + n * num_blocks_k;
    const float4* __restrict__ act_row_vec = (const float4* __restrict__)(activation + m * K);

    float sum = 0.0f;

    #pragma unroll 4
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0 w_block = w_row[kb];
        const float scale = half_to_float(w_block.d);

        const float4* act_vec = act_row_vec + kb * 8;

        float4 v0 = __ldg(&act_vec[0]);
        float4 v1 = __ldg(&act_vec[1]);
        float4 v2 = __ldg(&act_vec[2]);
        float4 v3 = __ldg(&act_vec[3]);
        float4 v4 = __ldg(&act_vec[4]);
        float4 v5 = __ldg(&act_vec[5]);
        float4 v6 = __ldg(&act_vec[6]);
        float4 v7 = __ldg(&act_vec[7]);

        float b0 = (float)((int8_t)(w_block.qs[0] & 0x0F) - 8) * v0.x
                 + (float)((int8_t)(w_block.qs[0] >> 4) - 8) * v4.y;
        float b1 = (float)((int8_t)(w_block.qs[1] & 0x0F) - 8) * v0.y
                 + (float)((int8_t)(w_block.qs[1] >> 4) - 8) * v4.z;
        float b2 = (float)((int8_t)(w_block.qs[2] & 0x0F) - 8) * v0.z
                 + (float)((int8_t)(w_block.qs[2] >> 4) - 8) * v4.w;
        float b3 = (float)((int8_t)(w_block.qs[3] & 0x0F) - 8) * v0.w
                 + (float)((int8_t)(w_block.qs[3] >> 4) - 8) * v5.x;
        float b4 = (float)((int8_t)(w_block.qs[4] & 0x0F) - 8) * v1.x
                 + (float)((int8_t)(w_block.qs[4] >> 4) - 8) * v5.y;
        float b5 = (float)((int8_t)(w_block.qs[5] & 0x0F) - 8) * v1.y
                 + (float)((int8_t)(w_block.qs[5] >> 4) - 8) * v5.z;
        float b6 = (float)((int8_t)(w_block.qs[6] & 0x0F) - 8) * v1.z
                 + (float)((int8_t)(w_block.qs[6] >> 4) - 8) * v5.w;
        float b7 = (float)((int8_t)(w_block.qs[7] & 0x0F) - 8) * v1.w
                 + (float)((int8_t)(w_block.qs[7] >> 4) - 8) * v6.x;
        float b8 = (float)((int8_t)(w_block.qs[8] & 0x0F) - 8) * v2.x
                 + (float)((int8_t)(w_block.qs[8] >> 4) - 8) * v6.y;
        float b9 = (float)((int8_t)(w_block.qs[9] & 0x0F) - 8) * v2.y
                 + (float)((int8_t)(w_block.qs[9] >> 4) - 8) * v6.z;
        float b10 = (float)((int8_t)(w_block.qs[10] & 0x0F) - 8) * v2.z
                  + (float)((int8_t)(w_block.qs[10] >> 4) - 8) * v6.w;
        float b11 = (float)((int8_t)(w_block.qs[11] & 0x0F) - 8) * v2.w
                  + (float)((int8_t)(w_block.qs[11] >> 4) - 8) * v7.x;
        float b12 = (float)((int8_t)(w_block.qs[12] & 0x0F) - 8) * v3.x
                  + (float)((int8_t)(w_block.qs[12] >> 4) - 8) * v7.y;
        float b13 = (float)((int8_t)(w_block.qs[13] & 0x0F) - 8) * v3.y
                  + (float)((int8_t)(w_block.qs[13] >> 4) - 8) * v7.z;
        float b14 = (float)((int8_t)(w_block.qs[14] & 0x0F) - 8) * v3.z
                  + (float)((int8_t)(w_block.qs[14] >> 4) - 8) * v7.w;
        float b15 = (float)((int8_t)(w_block.qs[15] & 0x0F) - 8) * v3.w
                  + (float)((int8_t)(w_block.qs[15] >> 4) - 8) * v4.x;

        sum += scale * (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 +
                        b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M <= 8) {
        dim3 block(64);
        dim3 grid((N + 63) / 64, M);
        gemm_q4_0_fp32_prefetch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_q4_0_fp32_large_batch_v11<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM with prefetching");
}
