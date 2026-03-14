/**
 * Final Optimized Quantized GEMM for DeepSeek-V2 MoE Down Projection
 * Q4_1 Weight x FP32 Activation with DP4A INT8 acceleration
 *
 * Parameters:
 *   - N = 5120 (output dimension)
 *   - K = 12288 (hidden dimension)
 *   - M = batch size (variable)
 *
 * This version combines the best-performing kernels:
 *   - v6 warp kernel for M <= 15 (best for small batch)
 *   - v7 tiled kernel for M >= 16 (best for large batch)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK4_1 32
#define QK8_1 32
#define WARP_SIZE 32

// Tiling parameters
#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define THREADS_M 8
#define THREADS_N 32
#define BATCH_THRESHOLD 16

// Q4_1 block structure
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

// ============================================================================
// Kernel 1: Warp-level (best for small batch M <= 15)
// From v6 - achieved 2.22 TFLOPS for M=1
// ============================================================================
__global__ void __launch_bounds__(256)
gemm_q4_1_warp_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    const int num_blocks = K / QK4_1;

    // Each warp processes multiple (m, n) pairs
    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int row = idx / N;
        const int col = idx % N;

        float sum = 0.0f;

        // Each lane processes a subset of K blocks
        for (int b = lane_id; b < num_blocks; b += WARP_SIZE) {
            const block_q4_1* w_block = &weight[col * num_blocks + b];
            float d_w = read_half_as_float(w_block->d);
            float m_w = read_half_as_float(w_block->m);

            int k_start = b * QK4_1;
            const float* act_ptr = &activation[row * K + k_start];

            // Load 32 activation values with vectorization
            float a_block[32];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
                a_block[i * 4] = a4.x;
                a_block[i * 4 + 1] = a4.y;
                a_block[i * 4 + 2] = a4.z;
                a_block[i * 4 + 3] = a4.w;
            }

            // Q8_1 quantization
            float a_max = 0.0f;
            float a_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK8_1; ++i) {
                a_max = fmaxf(a_max, fabsf(a_block[i]));
                a_sum += a_block[i];
            }

            float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
            float s_a = a_sum;

            // Quantize activation to INT8 and pack for DP4A
            int32_t a_packed[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);

                a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                              ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
            }

            // INT8 dot product with DP4A
            int32_t sumi = 0;

            // Process low nibbles (positions 0-15)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[i * 4]);

                int8_t w0 = (int8_t)(w_packed & 0x0F);
                int8_t w1 = (int8_t)((w_packed >> 8) & 0x0F);
                int8_t w2 = (int8_t)((w_packed >> 16) & 0x0F);
                int8_t w3 = (int8_t)((w_packed >> 24) & 0x0F);

                int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                            ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);

                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            // Process high nibbles (positions 16-31)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[i * 4]);

                int8_t w0 = (int8_t)((w_packed >> 4) & 0x0F);
                int8_t w1 = (int8_t)((w_packed >> 12) & 0x0F);
                int8_t w2 = (int8_t)((w_packed >> 20) & 0x0F);
                int8_t w3 = (int8_t)((w_packed >> 28) & 0x0F);

                int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                            ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);

                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            // Q4_1 x Q8_1 formula: output = d_w * d_a * sumi + m_w * s_a
            sum += d_w * d_a * (float)sumi + m_w * s_a;
        }

        // Warp reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            output[row * N + col] = sum;
        }
    }
}

// ============================================================================
// Kernel 2: Tiled with shared memory (best for large batch M >= 16)
// From v7 - achieved 14.37 TFLOPS for M=512
// ============================================================================
__global__ void __launch_bounds__(256)
gemm_q4_1_tiled_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;

    const int tid = threadIdx.y * THREADS_N + threadIdx.x;
    const int thread_m = threadIdx.y;
    const int thread_n = threadIdx.x;

    // Shared memory
    __shared__ float smem_activation[TILE_M][TILE_K];
    __shared__ int8_t smem_a_quantized[TILE_M][TILE_K];
    __shared__ float smem_a_scale[TILE_M];
    __shared__ float smem_a_sum[TILE_M];
    __shared__ block_q4_1 smem_weight[TILE_N];

    const int items_per_thread_m = TILE_M / THREADS_M;  // 8
    const int items_per_thread_n = TILE_N / THREADS_N;  // 4

    float accum[8][4];
    #pragma unroll
    for (int i = 0; i < items_per_thread_m; ++i) {
        #pragma unroll
        for (int j = 0; j < items_per_thread_n; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_blocks = K / QK4_1;

    // Loop over K blocks
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * QK4_1;

        // Load activation tile
        const int act_loads = (TILE_M * TILE_K) / (THREADS_M * THREADS_N);
        #pragma unroll 4
        for (int load_idx = 0; load_idx < act_loads; ++load_idx) {
            const int flat_idx = tid + load_idx * (THREADS_M * THREADS_N);
            const int m_local = flat_idx / TILE_K;
            const int k_local = flat_idx % TILE_K;
            const int m_global = block_m * TILE_M + m_local;
            const int k_global = k_start + k_local;

            if (m_global < M && k_global < K && m_local < TILE_M && k_local < TILE_K) {
                smem_activation[m_local][k_local] = activation[m_global * K + k_global];
            } else if (m_local < TILE_M && k_local < TILE_K) {
                smem_activation[m_local][k_local] = 0.0f;
            }
        }

        __syncthreads();

        // Quantize activation - use warp reduction for each row
        if (thread_m < items_per_thread_m) {
            for (int m_offset = 0; m_offset < TILE_M / items_per_thread_m; ++m_offset) {
                const int m_local = thread_m + m_offset * items_per_thread_m;
                if (m_local >= TILE_M) continue;

                float local_max = 0.0f;
                float local_sum = 0.0f;

                // Each thread processes elements across K
                for (int k = thread_n; k < TILE_K; k += THREADS_N) {
                    float val = smem_activation[m_local][k];
                    local_max = fmaxf(local_max, fabsf(val));
                    local_sum += val;
                }

                // Warp reduction
                #pragma unroll
                for (int offset = THREADS_N / 2; offset > 0; offset /= 2) {
                    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
                    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
                }

                if (thread_n == 0) {
                    float d_a = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
                    smem_a_scale[m_local] = d_a;
                    smem_a_sum[m_local] = local_sum;
                }
            }
        }

        __syncthreads();

        // Quantize values
        #pragma unroll 4
        for (int load_idx = 0; load_idx < act_loads; ++load_idx) {
            const int flat_idx = tid + load_idx * (THREADS_M * THREADS_N);
            const int m_local = flat_idx / TILE_K;
            const int k_local = flat_idx % TILE_K;

            if (m_local < TILE_M && k_local < TILE_K) {
                float val = smem_activation[m_local][k_local];
                float d_a = smem_a_scale[m_local];
                smem_a_quantized[m_local][k_local] = (int8_t)__float2int_rn(val / d_a);
            }
        }

        // Load weights
        for (int n_local = tid; n_local < TILE_N; n_local += (THREADS_M * THREADS_N)) {
            const int n_global = block_n * TILE_N + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[n_global * num_k_blocks + k_block];
            }
        }

        __syncthreads();

        // INT8 GEMM with DP4A
        #pragma unroll
        for (int i = 0; i < items_per_thread_m; ++i) {
            const int m_local = thread_m * items_per_thread_m + i;
            const int m_global = block_m * TILE_M + m_local;
            if (m_global >= M || m_local >= TILE_M) continue;

            float d_a = smem_a_scale[m_local];
            float s_a = smem_a_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_per_thread_n; ++j) {
                const int n_local = thread_n * items_per_thread_n + j;
                const int n_global = block_n * TILE_N + n_local;
                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                float d_w = read_half_as_float(w_block->d);
                float m_w = read_half_as_float(w_block->m);

                int32_t sumi = 0;

                // DP4A optimized dot product - low nibbles
                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_quantized[m_local][ii * 4]);

                    uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);
                    int8_t w0 = (int8_t)(w_packed & 0x0F);
                    int8_t w1 = (int8_t)((w_packed >> 8) & 0x0F);
                    int8_t w2 = (int8_t)((w_packed >> 16) & 0x0F);
                    int8_t w3 = (int8_t)((w_packed >> 24) & 0x0F);
                    int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                                ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                // High nibbles
                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_quantized[m_local][16 + ii * 4]);

                    uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);
                    int8_t w0 = (int8_t)((w_packed >> 4) & 0x0F);
                    int8_t w1 = (int8_t)((w_packed >> 12) & 0x0F);
                    int8_t w2 = (int8_t)((w_packed >> 20) & 0x0F);
                    int8_t w3 = (int8_t)((w_packed >> 28) & 0x0F);
                    int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                                ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                accum[i][j] += d_w * d_a * (float)sumi + m_w * s_a;
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < items_per_thread_m; ++i) {
        const int m_global = block_m * TILE_M + thread_m * items_per_thread_m + i;
        if (m_global >= M) continue;

        #pragma unroll
        for (int j = 0; j < items_per_thread_n; ++j) {
            const int n_global = block_n * TILE_N + thread_n * items_per_thread_n + j;
            if (n_global < N) {
                output[m_global * N + n_global] = accum[i][j];
            }
        }
    }
}

// ============================================================================
// Host function with strategy dispatch
// ============================================================================
torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M < BATCH_THRESHOLD) {
        // Small batch: warp-level kernel (best for M <= 15)
        int total_outputs = M * N;
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_blocks_grid = (total_outputs + warps_per_block - 1) / warps_per_block;

        gemm_q4_1_warp_kernel<<<num_blocks_grid, threads_per_block>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: tiled kernel (best for M >= 16)
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks(
            (M + TILE_M - 1) / TILE_M,
            (N + TILE_N - 1) / TILE_N
        );

        gemm_q4_1_tiled_kernel<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 x Q8_1 GEMM - Final Optimized");
}
