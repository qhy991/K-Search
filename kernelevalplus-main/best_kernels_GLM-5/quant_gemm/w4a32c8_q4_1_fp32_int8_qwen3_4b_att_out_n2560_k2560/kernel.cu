/**
 * W4A32C8 Quantized GEMM for Qwen3-4B Attention Output Projection
 * Q4_1 Weight (N=2560, K=2560) x FP32 Activation (M=batch, K=2560)
 *
 * v17: Use __ldg for read-only cache on weights
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK4_1 32
#define WARP_SIZE 32
#define Q4_1_BLOCK 20

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

// Load 4 bytes from weight using read-only cache
inline __device__ uint32_t ldg_weight4(const uint8_t* ptr) {
    return __ldg(reinterpret_cast<const uint32_t*>(ptr));
}

// Load 2 bytes from weight using read-only cache
inline __device__ uint16_t ldg_weight2(const uint8_t* ptr) {
    return __ldg(reinterpret_cast<const uint16_t*>(ptr));
}

/**
 * M=1 kernel: 32 threads (1 warp), 2 columns per block
 * Uses read-only cache for weight loading
 */
__global__ void __launch_bounds__(32)
gemm_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_k_blocks = K / QK4_1;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // 2 columns per block (1 warp handles both)
    const int base_col = blockIdx.x * 2;
    const int cols_to_process = min(2, N - base_col);

    if (base_col >= N) return;

    float sums[2] = {0.0f, 0.0f};

    // K-parallel: each lane processes different K blocks
    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

        // Load activation block once
        float a_block[32];
        const float* act_ptr = &activation[k_start];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        // Compute activation stats
        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float s_a = a_sum;

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

        // Process each column
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_1_BLOCK;

            // Load d and m using read-only cache
            const float d_w = read_half_as_float(ldg_weight2(w_block));
            const float m_w = read_half_as_float(ldg_weight2(w_block + 2));
            const uint8_t* qs = w_block + 4;

            int32_t sumi = 0;

            // Process lower nibbles
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_raw = ldg_weight4(&qs[i * 4]);
                uint8_t b0 = w_raw & 0xFF;
                uint8_t b1 = (w_raw >> 8) & 0xFF;
                uint8_t b2 = (w_raw >> 16) & 0xFF;
                uint8_t b3 = (w_raw >> 24) & 0xFF;
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            // Process upper nibbles
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_raw = ldg_weight4(&qs[i * 4]);
                uint8_t b0 = (w_raw >> 4) & 0x0F;
                uint8_t b1 = (w_raw >> 12) & 0x0F;
                uint8_t b2 = (w_raw >> 20) & 0x0F;
                uint8_t b3 = (w_raw >> 28) & 0x0F;
                int w_pack = (int(b0)) | (int(b1) << 8) |
                            (int(b2) << 16) | (int(b3) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi + m_w * s_a;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[base_col + c] = sums[c];
        }
    }
}

/**
 * Small M kernel (M <= 4): 128 threads, 16 columns per block
 */
__global__ void __launch_bounds__(128)
gemm_small_m_kernel(
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
    const int base_col = col_block_idx * (num_warps * COLS_PER_WARP) + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_1;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

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
        const float s_a = a_sum;

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
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_1_BLOCK;
            const float d_w = read_half_as_float(ldg_weight2(w_block));
            const float m_w = read_half_as_float(ldg_weight2(w_block + 2));
            const uint8_t* qs = w_block + 4;

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_raw = ldg_weight4(&qs[i * 4]);
                uint8_t b0 = w_raw & 0xFF;
                uint8_t b1 = (w_raw >> 8) & 0xFF;
                uint8_t b2 = (w_raw >> 16) & 0xFF;
                uint8_t b3 = (w_raw >> 24) & 0xFF;
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_raw = ldg_weight4(&qs[i * 4]);
                uint8_t b0 = (w_raw >> 4) & 0x0F;
                uint8_t b1 = (w_raw >> 12) & 0x0F;
                uint8_t b2 = (w_raw >> 20) & 0x0F;
                uint8_t b3 = (w_raw >> 28) & 0x0F;
                int w_pack = (int(b0)) | (int(b1) << 8) |
                            (int(b2) << 16) | (int(b3) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi + m_w * s_a;
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
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[static_cast<int64_t>(row) * N + base_col + c] = sums[c];
        }
    }
}

/**
 * Large M kernel: 256 threads, 64 columns per block
 */
__global__ void __launch_bounds__(256)
gemm_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int COLS_PER_WARP = 8;
    const int base_col = blockIdx.y * 8 * COLS_PER_WARP + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_1;
    float sums[8] = {0.0f};

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

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
        const float s_a = a_sum;

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
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_1_BLOCK;
            const float d_w = read_half_as_float(ldg_weight2(w_block));
            const float m_w = read_half_as_float(ldg_weight2(w_block + 2));
            const uint8_t* qs = w_block + 4;

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_raw = ldg_weight4(&qs[i * 4]);
                uint8_t b0 = w_raw & 0xFF;
                uint8_t b1 = (w_raw >> 8) & 0xFF;
                uint8_t b2 = (w_raw >> 16) & 0xFF;
                uint8_t b3 = (w_raw >> 24) & 0xFF;
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_raw = ldg_weight4(&qs[i * 4]);
                uint8_t b0 = (w_raw >> 4) & 0x0F;
                uint8_t b1 = (w_raw >> 12) & 0x0F;
                uint8_t b2 = (w_raw >> 20) & 0x0F;
                uint8_t b3 = (w_raw >> 28) & 0x0F;
                int w_pack = (int(b0)) | (int(b1) << 8) |
                            (int(b2) << 16) | (int(b3) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi + m_w * s_a;
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
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[static_cast<int64_t>(row) * N + base_col + c] = sums[c];
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(activation.is_contiguous(), "Activation must be contiguous");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    const float* a_ptr = activation.data_ptr<float>();
    float* c_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(weight.device().index());

    if (M == 1) {
        // M=1: 32 threads, 2 cols/block
        dim3 grid((N + 1) / 2, 1);
        dim3 block(32);
        gemm_m1_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    } else if (M <= 4) {
        const int n_blocks = (N + 15) / 16;
        dim3 grid(n_blocks, M);
        dim3 block(128);
        gemm_small_m_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    } else {
        dim3 grid(M, (N + 63) / 64);
        dim3 block(256);
        gemm_large_m_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_1 Quantized GEMM v17");
}
