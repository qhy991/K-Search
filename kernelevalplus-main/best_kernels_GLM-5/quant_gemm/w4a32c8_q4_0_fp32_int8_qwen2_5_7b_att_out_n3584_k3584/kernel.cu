/**
 * Final Optimized Quantized GEMM Kernel for Qwen2.5-7B Attention Output
 * - N: 3584, K: 3584
 * - Strategy: Best configuration from v15
 * - Direct FP32 computation with optimal thread configuration
 *
 * Performance:
 *   M=1:   493 GFLOPS (12.2% of baseline)
 *   M=512: 3128 GFLOPS
 *
 * Note: GGML baseline achieves 4030 GFLOPS at M=1 through different
 * memory access patterns or caching strategies not replicated here.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int QK = 32;
constexpr int Q4_0_BLOCK_SIZE = 18;

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

// ============================================================================
// Small M kernel (M <= 32): 128 threads, one output per thread
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q4_0_small_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const int num_k_blocks = K / QK;
    const float* __restrict__ act_row = activation + m * K;

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_start = kb * QK;
        const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK_SIZE;

        float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* w_qs = w_block + 2;

        float block_sum = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_qs[i];
            float w0 = static_cast<float>(packed & 0x0F);
            float w1 = static_cast<float>((packed >> 4) & 0x0F);

            float a0 = act_row[k_start + i];
            float a1 = act_row[k_start + i + 16];

            block_sum += w0 * a0 + w1 * a1;
            act_sum += a0 + a1;
        }

        sum += w_scale * (block_sum - 8.0f * act_sum);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Medium M kernel (32 < M <= 256): 64 threads for higher occupancy
// ============================================================================
__global__ void __launch_bounds__(64) gemm_q4_0_medium_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const int num_k_blocks = K / QK;
    const float* __restrict__ act_row = activation + m * K;

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_start = kb * QK;
        const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK_SIZE;

        float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* w_qs = w_block + 2;

        float block_sum = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_qs[i];
            float w0 = static_cast<float>(packed & 0x0F);
            float w1 = static_cast<float>((packed >> 4) & 0x0F);

            float a0 = act_row[k_start + i];
            float a1 = act_row[k_start + i + 16];

            block_sum += w0 * a0 + w1 * a1;
            act_sum += a0 + a1;
        }

        sum += w_scale * (block_sum - 8.0f * act_sum);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Large M kernel (M > 256): 2D tiling for compute-bound regime
// ============================================================================
template<int TILE_M, int TILE_N>
__global__ void __launch_bounds__(TILE_M * TILE_N) gemm_q4_0_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n = blockIdx.x * TILE_N + tx;
    const int m = blockIdx.y * TILE_M + ty;

    const int num_k_blocks = K / QK;
    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_start = kb * QK;

        float block_sum = 0.0f;
        float act_sum = 0.0f;

        if (n < N && m < M) {
            const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK_SIZE;
            float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
            const uint8_t* w_qs = w_block + 2;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = w_qs[i];
                float w0 = static_cast<float>(packed & 0x0F);
                float w1 = static_cast<float>((packed >> 4) & 0x0F);

                float a0 = activation[m * K + k_start + i];
                float a1 = activation[m * K + k_start + i + 16];

                block_sum += w0 * a0 + w1 * a1;
                act_sum += a0 + a1;
            }

            sum += w_scale * (block_sum - 8.0f * act_sum);
        }
    }

    if (n < N && m < M) {
        output[m * N + n] = sum;
    }
}

// ============================================================================
// Host function with strategy dispatch
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M <= 32) {
        // Small M: Memory-bound, use 128 threads
        dim3 block(128);
        dim3 grid((N + 127) / 128, M);

        gemm_q4_0_small_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 256) {
        // Medium M: Transitional, use 64 threads
        dim3 block(64);
        dim3 grid((N + 63) / 64, M);

        gemm_q4_0_medium_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large M: Compute-bound, use 2D tiling
        const int TILE_M = 8;
        const int TILE_N = 32;

        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        gemm_q4_0_large_m_kernel<TILE_M, TILE_N><<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 GEMM Final for Qwen2.5-7B");
}
