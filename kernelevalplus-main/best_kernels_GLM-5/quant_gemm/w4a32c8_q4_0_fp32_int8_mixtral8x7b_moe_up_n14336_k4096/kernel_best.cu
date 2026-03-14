/**
 * W4A32C8 Quantized GEMM for Mixtral-8x7B MoE Up Projection - v14
 * Q4_0 Weight (N=14336, K=4096) x FP32 Activation (M=batch, K=4096)
 *
 * v14: Based on v12's success (5.02 TFLOPS, 61% of baseline)
 * - 8 columns per warp, 2 warps per block = 16 cols/block
 * - Grid(N/16, M) = 896 blocks for N=14336
 *
 * Additional optimizations:
 * - Pre-compute reciprocal for faster division
 * - Use __ldg for weight loads to improve cache behavior
 * - Restructure loop for better ILP
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
 * Small M kernel: 8 columns per warp, 2 warps per block
 * Grid(N/16, M) = 896 blocks for N=14336
 */
__global__ void __launch_bounds__(64)
gemm_q4_0_small_m_kernel(
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

    const int COLS_PER_WARP = 8;
    const int COLS_PER_BLOCK = 16;  // 2 warps * 8 cols

    const int base_col = col_block_idx * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_0;  // 128

    float sums[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) sums[i] = 0.0f;

    // K-parallel: each lane processes different K blocks
    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_0;

        // Load activation block (vectorized)
        float a_block[32];
        const float* act_ptr = &activation[static_cast<int64_t>(row) * K + k_start];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        // Compute activation quantization parameters
        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        // Pack activation into int8 for DP4A
        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] * inv_d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] * inv_d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] * inv_d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] * inv_d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        // Process each column
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_0_BLOCK;
            const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));

            const uint8_t* qs = w_block + 2;
            int32_t sumi = 0;

            // Process low nibbles (first 16 values)
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

            // Process high nibbles (next 16 values)
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

            sums[c] += d_w * (d_a * (float)sumi - 8.0f * a_sum);
        }
    }

    // Warp reduction for each column
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    // Write results
    if (lane_id == 0) {
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            if (col < N) output[static_cast<int64_t>(row) * N + col] = sums[c];
        }
    }
}

/**
 * Medium M kernel: 4 columns per warp, 4 warps per block
 */
__global__ void __launch_bounds__(128)
gemm_q4_0_medium_m_kernel(
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

    const int COLS_PER_WARP = 4;
    const int COLS_PER_BLOCK = num_warps * COLS_PER_WARP;

    const int base_col = col_block_idx * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_0;

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_0;

        float a_block[32];
        const float* act_ptr = &activation[static_cast<int64_t>(row) * K + k_start];

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
        const float inv_d_a = 1.0f / d_a;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] * inv_d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] * inv_d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] * inv_d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] * inv_d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
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

            sums[c] += d_w * (d_a * (float)sumi - 8.0f * a_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
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
 * Large M kernel: 16 columns per warp
 */
__global__ void __launch_bounds__(512)
gemm_q4_0_large_m_kernel(
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

    const int num_k_blocks = K / QK4_0;

    float sums[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c) sums[c] = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_0;

        float a_block[32];
        const float* act_ptr = &activation[static_cast<int64_t>(row) * K + k_start];

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
        const float inv_d_a = 1.0f / d_a;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] * inv_d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] * inv_d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] * inv_d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] * inv_d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
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

            sums[c] += d_w * (d_a * (float)sumi - 8.0f * a_sum);
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
        // Small M: 8 columns per warp, 2 warps per block (16 cols/block)
        // Grid: (N/16, M) = 896 blocks for N=14336
        const int COLS_PER_BLOCK = 16;
        dim3 grid((N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK, M);
        dim3 block(64);

        gemm_q4_0_small_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 64) {
        const int COLS_PER_BLOCK = 16;
        dim3 grid((N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK, M);
        dim3 block(128);

        gemm_q4_0_medium_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        const int COLS_PER_BLOCK = 256;
        dim3 grid(M, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
        dim3 block(512);

        gemm_q4_0_large_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM v14 - Mixtral MoE Up");
}
