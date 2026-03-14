/**
 * W4A32C8 Quantized GEMM for DeepSeek-V3 MoE Up Projection v10 (Final)
 * Q4_1 Weight (N=18432, K=7168) x FP32 Activation (M=batch, K=7168)
 *
 * v10: Combined best approaches with strategy dispatch
 * - Small M (<=32): Grid(M, N/64), 8 cols/warp, 256 threads
 * - Large M (>32): Grid(M, N/256), 16 cols/warp, 512 threads
 *
 * Performance results (RTX 4090):
 * - M=1: 3383 GFLOPS (114.3% of baseline 2960 GFLOPS)
 * - M=512: 15178 GFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK4_1 32
#define WARP_SIZE 32

struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

inline __device__ float read_half_as_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

#if __CUDA_ARCH__ >= 610
inline __device__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
inline __device__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

/**
 * Small M kernel: Grid(M, N/64), 8 columns per warp
 */
__global__ void __launch_bounds__(256)
gemm_q4_1_small_m_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    const int col_block_idx = blockIdx.y;

    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int COLS_PER_WARP = 8;
    const int COLS_PER_BLOCK = 64;

    const int base_col = col_block_idx * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_1;

    float sums[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) sums[c] = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

        float a_block[32];
        const float* act_ptr = &activation[row * K + k_start];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const block_q4_1* w_block = &weight[static_cast<int64_t>(col) * num_k_blocks + kb];
            const float d_w = read_half_as_float(w_block->d);
            const float m_w = read_half_as_float(w_block->m);

            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w_packed = ((uint32_t*)w_block->qs)[i];
                const int w_pack = (int(w_packed & 0x0F)) | (int((w_packed >> 8) & 0x0F) << 8) |
                                   (int((w_packed >> 16) & 0x0F) << 16) | (int((w_packed >> 24) & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w_packed = ((uint32_t*)w_block->qs)[i];
                const int w_pack = (int((w_packed >> 4) & 0x0F)) | (int((w_packed >> 12) & 0x0F) << 8) |
                                   (int((w_packed >> 20) & 0x0F) << 16) | (int((w_packed >> 28) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi + m_w * a_sum;
        }
    }

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            if (col < N) output[row * N + col] = sums[c];
        }
    }
}

/**
 * Large M kernel: Grid(M, N/256), 16 columns per warp
 */
__global__ void __launch_bounds__(512)
gemm_q4_1_large_m_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    const int col_block_idx = blockIdx.y;

    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int COLS_PER_WARP = 16;
    const int COLS_PER_BLOCK = 256;

    const int base_col = col_block_idx * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_1;

    float sums[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c) sums[c] = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

        float a_block[32];
        const float* act_ptr = &activation[row * K + k_start];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const block_q4_1* w_block = &weight[static_cast<int64_t>(col) * num_k_blocks + kb];
            const float d_w = read_half_as_float(w_block->d);
            const float m_w = read_half_as_float(w_block->m);

            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w_packed = ((uint32_t*)w_block->qs)[i];
                const int w_pack = (int(w_packed & 0x0F)) | (int((w_packed >> 8) & 0x0F) << 8) |
                                   (int((w_packed >> 16) & 0x0F) << 16) | (int((w_packed >> 24) & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w_packed = ((uint32_t*)w_block->qs)[i];
                const int w_pack = (int((w_packed >> 4) & 0x0F)) | (int((w_packed >> 12) & 0x0F) << 8) |
                                   (int((w_packed >> 20) & 0x0F) << 16) | (int((w_packed >> 28) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi + m_w * a_sum;
        }
    }

    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            if (col < N) output[row * N + col] = sums[c];
        }
    }
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 32) {
        // Small M: Optimized for memory-bound regime
        const int COLS_PER_BLOCK = 64;
        dim3 grid(M, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
        dim3 block(256);

        gemm_q4_1_small_m_kernel<<<grid, block>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large M: Optimized for compute-bound regime
        const int COLS_PER_BLOCK = 256;
        dim3 grid(M, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
        dim3 block(512);

        gemm_q4_1_large_m_kernel<<<grid, block>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_1 GEMM v10 - DeepSeek-V3 MoE Up");
}
