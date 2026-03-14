/**
 * W4A32C8 Quantized GEMM for Qwen3-1.5B Attention QKV Projection
 * Q4_0 Weight (N=7168, K=2048) x FP32 Activation (M=batch, K=2048)
 *
 * v7: Warp-parallel K-block processing with inline assembly DP4A
 * Based on deepseek_v3_moe_up pattern (4671 GFLOPS achieved)
 *
 * Key optimizations:
 * - Inline assembly DP4A for INT8 dot product
 * - K-parallel within warp: each lane processes different K blocks
 * - Warp-level reduction using __shfl_down_sync
 * - Grid(N/16, M) for better SM utilization on small M
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK4_0 32
#define WARP_SIZE 32
#define Q4_0_BLOCK 18

inline __device__ float read_half_as_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

// Inline assembly DP4A - critical for performance
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
 * Optimized kernel with warp-parallel K-block processing
 * Grid: (N/16, M) -> for N=7168, M=1: grid(448, 1) = 448 blocks
 * Each block: 128 threads = 4 warps, each warp handles 4 columns
 * Within warp: lanes (threads) process different K blocks in parallel
 */
__global__ void __launch_bounds__(128)
gemm_q4_0_warp_parallel_kernel(
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
    const int num_warps = blockDim.x / WARP_SIZE;  // 4

    const int COLS_PER_WARP = 4;
    const int COLS_PER_BLOCK = num_warps * COLS_PER_WARP;  // 16

    const int base_col = col_block_idx * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_0;

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // K-parallel: each lane processes different K blocks
    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_0;

        // Load activation block (32 values)
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

        // Q8_1 quantization: compute scale and sum
        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        // Pack activation into int32 for DP4A
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

        // Process each column this warp handles
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_0_BLOCK;
            const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));

            const uint8_t* qs = w_block + 2;
            int32_t sumi = 0;

            // Process lower nibbles (0-15, positions 0-15)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0];
                uint8_t b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2];
                uint8_t b3 = qs[i * 4 + 3];
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            // Process upper nibbles (0-15, positions 16-31)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0];
                uint8_t b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2];
                uint8_t b3 = qs[i * 4 + 3];
                int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                            (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            // Apply llama.cpp formula: d_w * (d_a * sumi - 8.0 * a_sum)
            sums[c] += d_w * (d_a * (float)sumi - 8.0f * a_sum);
        }
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    // Write results (only lane 0 writes for each column)
    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[row * N + base_col + c] = sums[c];
        }
    }
}

/**
 * Medium M kernel: 1D strided with optimized thread block
 */
__global__ void __launch_bounds__(256)
gemm_q4_0_medium_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int n_stride = blockDim.x;
    const float* __restrict__ a_row = activation + row * K;
    const int num_k_blocks = K / QK4_0;

    for (int n = tid; n < N; n += n_stride) {
        float acc = 0.0f;

        for (int kb = 0; kb < num_k_blocks; ++kb) {
            const int k_start = kb * QK4_0;

            float a_block[32];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4 a4 = *reinterpret_cast<const float4*>(&a_row[k_start + i * 4]);
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

            const uint8_t* w_block = weight + (static_cast<int64_t>(n) * num_k_blocks + kb) * Q4_0_BLOCK;
            const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
            const uint8_t* qs = w_block + 2;

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0];
                uint8_t b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2];
                uint8_t b3 = qs[i * 4 + 3];
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0];
                uint8_t b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2];
                uint8_t b3 = qs[i * 4 + 3];
                int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                            (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            acc += d_w * (d_a * (float)sumi - 8.0f * a_sum);
        }

        output[row * N + n] = acc;
    }
}

/**
 * Large M kernel: 2D grid
 */
__global__ void __launch_bounds__(256)
gemm_q4_0_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.y * 16 + threadIdx.y;
    const int col = blockIdx.x * 16 + threadIdx.x;

    if (row >= M || col >= N) return;

    const float* __restrict__ a_row = activation + row * K;
    const int num_k_blocks = K / QK4_0;
    float acc = 0.0f;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_start = kb * QK4_0;

        float a_block[32];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&a_row[k_start + i * 4]);
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

        const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_0_BLOCK;
        const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* qs = w_block + 2;

        int32_t sumi = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t b0 = qs[i * 4 + 0];
            uint8_t b1 = qs[i * 4 + 1];
            uint8_t b2 = qs[i * 4 + 2];
            uint8_t b3 = qs[i * 4 + 3];
            int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                        (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
            sumi = dp4a(a_packed[i], w_pack, sumi);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t b0 = qs[i * 4 + 0];
            uint8_t b1 = qs[i * 4 + 1];
            uint8_t b2 = qs[i * 4 + 2];
            uint8_t b3 = qs[i * 4 + 3];
            int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                        (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);
            sumi = dp4a(a_packed[i + 4], w_pack, sumi);
        }

        acc += d_w * (d_a * (float)sumi - 8.0f * a_sum);
    }

    output[row * N + col] = acc;
}

// PyTorch interface with strategy dispatch
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(activation.is_contiguous(), "Activation must be contiguous");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    const float* a_ptr = activation.data_ptr<float>();
    float* c_ptr = output.data_ptr<float>();

    if (M <= 8) {
        // Small M: warp-parallel K-block processing
        // Grid: (N/16, M) -> for N=7168, M=1: grid(448, 1)
        const int n_blocks = (N + 15) / 16;
        const dim3 blocks(n_blocks, M);

        gemm_q4_0_warp_parallel_kernel<<<blocks, 128>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    } else if (M <= 128) {
        // Medium M: 1D strided
        gemm_q4_0_medium_m_kernel<<<M, 256>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    } else {
        // Large M: 2D grid
        const dim3 threads(16, 16);
        const dim3 blocks((N + 15) / 16, (M + 15) / 16);

        gemm_q4_0_large_m_kernel<<<blocks, threads>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (Q4_0 x Q8_1 with warp-parallel DP4A)");
}
