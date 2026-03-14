/**
 * W4A32C8 Quantized GEMM V9 - Fine-tuned from best configuration
 * 
 * Based on Final version which achieved 5.03 TFLOPS (54.6% of baseline)
 * 
 * Optimizations:
 * 1. Use __ldg for read-only cache hints on weights
 * 2. Reduce register pressure with __restrict__
 * 3. Fine-tune grid/block configuration
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK4_1 32
#define WARP_SIZE 32
#define Q4_1_BLOCK 20

inline __device__ float __half_to_float(uint16_t h) {
    return __half2float(__ushort_as_half(h));
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
 * Small M kernel: 8 cols/warp, 2 warps/block (same as final but with __ldg)
 */
__global__ void __launch_bounds__(64)
gemm_q4_1_small_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.y;
    const int col_block_idx = blockIdx.x;

    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int COLS_PER_WARP = 8;
    const int COLS_PER_BLOCK = num_warps * COLS_PER_WARP;

    const int base_col = col_block_idx * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_1;

    float sums[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) sums[c] = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

        // Load activation with __ldg for read-only cache
        float a_block[32];
        const float* act_ptr = activation + static_cast<int64_t>(row) * K + k_start;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4* ptr = reinterpret_cast<const float4*>(act_ptr + i * 4);
            float4 a4 = __ldg(ptr);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        // Quantize activation
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_packed[8];
        int32_t s_a = 0;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = __float2int_rn(a_block[i * 4 + 0] / d_a);
            const int8_t q1 = __float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = __float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = __float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
            s_a += q0 + q1 + q2 + q3;
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_1_BLOCK;

            // Use __ldg for weight metadata
            const uint16_t d_w_half = __ldg(reinterpret_cast<const uint16_t*>(w_block));
            const uint16_t m_w_half = __ldg(reinterpret_cast<const uint16_t*>(w_block + 2));
            const float d_w = __half_to_float(d_w_half);
            const float m_w = __half_to_float(m_w_half);

            const uint8_t* qs = w_block + 4;
            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4* qs_ptr = reinterpret_cast<const uint4*>(qs + i * 4);
                uint4 b = __ldg(qs_ptr);
                uint8_t b0 = b.x & 0xFF;
                uint8_t b1 = (b.x >> 8) & 0xFF;
                uint8_t b2 = (b.x >> 16) & 0xFF;
                uint8_t b3 = (b.x >> 24) & 0xFF;
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4* qs_ptr = reinterpret_cast<const uint4*>(qs + i * 4);
                uint4 b = __ldg(qs_ptr);
                uint8_t b0 = b.x & 0xFF;
                uint8_t b1 = (b.x >> 8) & 0xFF;
                uint8_t b2 = (b.x >> 16) & 0xFF;
                uint8_t b3 = (b.x >> 24) & 0xFF;
                int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                            (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_a * (d_w * (float)sumi + m_w * (float)s_a);
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
            if (col < N) output[static_cast<int64_t>(row) * N + col] = sums[c];
        }
    }
}

/**
 * Large M kernel: 16 cols/warp, 16 warps per block
 */
__global__ void __launch_bounds__(512)
gemm_q4_1_large_m_kernel(
    const uint8_t* __restrict__ weight,
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
        const float* act_ptr = activation + static_cast<int64_t>(row) * K + k_start;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(act_ptr + i * 4);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_packed[8];
        int32_t s_a = 0;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = __float2int_rn(a_block[i * 4 + 0] / d_a);
            const int8_t q1 = __float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = __float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = __float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
            s_a += q0 + q1 + q2 + q3;
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_1_BLOCK;

            const float d_w = __half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
            const float m_w = __half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));

            const uint8_t* qs = w_block + 4;
            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0], b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2], b3 = qs[i * 4 + 3];
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0], b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2], b3 = qs[i * 4 + 3];
                int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                            (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_a * (d_w * (float)sumi + m_w * (float)s_a);
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
            if (col < N) output[static_cast<int64_t>(row) * N + col] = sums[c];
        }
    }
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 8) {
        const int COLS_PER_BLOCK = 16;
        dim3 grid((N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK, M);
        dim3 block(64);

        gemm_q4_1_small_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        const int COLS_PER_BLOCK = 256;
        dim3 grid(M, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
        dim3 block(512);

        gemm_q4_1_large_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM Final - Mixtral MoE Up");
}
