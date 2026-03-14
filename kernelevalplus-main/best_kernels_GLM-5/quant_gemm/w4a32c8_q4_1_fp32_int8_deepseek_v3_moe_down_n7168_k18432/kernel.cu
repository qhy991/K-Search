/**
 * Optimized Quantized GEMM for DeepSeek-V3 MoE Down Projection
 * Q4_1 Weight x FP32 Activation with DP4A INT8 acceleration
 *
 * Version 12 - Shared memory caching for activation, multiple outputs per warp
 *
 * Strategy: Each block processes multiple output elements sharing the same row.
 * Use shared memory to cache the quantized activation for reuse.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK4_1 32
#define QK8_1 32
#define WARP_SIZE 32

struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

inline __device__ float read_half_as_float(uint16_t h) {
    return __half2float(reinterpret_cast<half&>(h));
}

inline __device__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

// Process 8 outputs per warp (each thread handles 1/4 of K)
#define OUTPUTS_PER_WARP 8
#define K_BLOCKS_PER_THREAD 18  // 576 / 32 = 18

__global__ void __launch_bounds__(256)
gemm_q4_1_multi_output_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    // Each warp handles OUTPUTS_PER_WARP consecutive outputs in the same row
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    const int num_k_blocks = K / QK4_1;  // 576

    // Each warp processes multiple output groups
    const int total_groups = M * ((N + OUTPUTS_PER_WARP - 1) / OUTPUTS_PER_WARP);

    for (int group = warp_id; group < total_groups; group += num_warps) {
        const int row = group / ((N + OUTPUTS_PER_WARP - 1) / OUTPUTS_PER_WARP);
        const int col_start = (group % ((N + OUTPUTS_PER_WARP - 1) / OUTPUTS_PER_WARP)) * OUTPUTS_PER_WARP;

        if (row >= M) continue;

        // Shared memory for activation block
        __shared__ float s_act[32];  // One K block of activation
        __shared__ float s_scale;     // Q8_1 scale
        __shared__ float s_sum;       // Q8_1 sum
        __shared__ int32_t s_a_packed[8];  // Quantized activation

        float results[OUTPUTS_PER_WARP];
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_WARP; ++i) {
            results[i] = 0.0f;
        }

        // Process K dimension
        for (int kb = 0; kb < num_k_blocks; ++kb) {
            const int k_start = kb * QK4_1;

            // Load and quantize activation (only first warp does this)
            if (lane_id < 32) {
                const float* act_ptr = &activation[row * K + k_start + lane_id];
                s_act[lane_id] = act_ptr[0];
            }
            __syncwarp();

            // Compute scale and sum
            float local_max = fabsf(s_act[lane_id]);
            float local_sum = s_act[lane_id];

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            }

            if (lane_id == 0) {
                s_scale = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
                s_sum = local_sum;
            }
            __syncwarp();

            // Quantize activation
            if (lane_id < 32) {
                int8_t q = (int8_t)__float2int_rn(s_act[lane_id] / s_scale);
                // Pack 4 values into int32
                int packed = 0;
                packed |= (int(__shfl_sync(0xffffffff, q, lane_id * 4 + 0)) & 0xFF);
                packed |= (int(__shfl_sync(0xffffffff, q, lane_id * 4 + 1)) & 0xFF) << 8;
                packed |= (int(__shfl_sync(0xffffffff, q, lane_id * 4 + 2)) & 0xFF) << 16;
                packed |= (int(__shfl_sync(0xffffffff, q, lane_id * 4 + 3)) & 0xFF) << 24;

                if (lane_id < 8) {
                    s_a_packed[lane_id] = packed;
                }
            }
            __syncwarp();

            // Each thread processes 1-2 output columns
            #pragma unroll
            for (int out_idx = 0; out_idx < OUTPUTS_PER_WARP; ++out_idx) {
                const int col = col_start + out_idx;
                if (col >= N) continue;

                const block_q4_1* w_col = &weight[col * num_k_blocks + kb];
                const float d_w = read_half_as_float(w_col->d);
                const float m_w = read_half_as_float(w_col->m);

                int32_t sumi = 0;

                // Load weight and compute DP4A
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_col->qs[i * 4]);

                    // Low nibbles
                    int8_t w0 = (int8_t)(w_packed & 0x0F);
                    int8_t w1 = (int8_t)((w_packed >> 8) & 0x0F);
                    int8_t w2 = (int8_t)((w_packed >> 16) & 0x0F);
                    int8_t w3 = (int8_t)((w_packed >> 24) & 0x0F);
                    int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                                ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                    sumi = dp4a(s_a_packed[i], w_pack, sumi);

                    // High nibbles
                    w0 = (int8_t)((w_packed >> 4) & 0x0F);
                    w1 = (int8_t)((w_packed >> 12) & 0x0F);
                    w2 = (int8_t)((w_packed >> 20) & 0x0F);
                    w3 = (int8_t)((w_packed >> 28) & 0x0F);
                    w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                            ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                    sumi = dp4a(s_a_packed[i + 4], w_pack, sumi);
                }

                results[out_idx] += d_w * s_scale * (float)sumi + m_w * s_sum;
            }
        }

        // Write results
        #pragma unroll
        for (int out_idx = 0; out_idx < OUTPUTS_PER_WARP; ++out_idx) {
            const int col = col_start + out_idx;
            if (col < N) {
                output[row * N + col] = results[out_idx];
            }
        }
    }
}

/**
 * Simple warp kernel (original approach, proven performance)
 */
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

    const int num_k_blocks = K / QK4_1;

    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int row = idx / N;
        const int col = idx % N;

        if (row >= M) continue;

        float sum = 0.0f;
        const float* act_row = &activation[row * K];
        const block_q4_1* w_col = &weight[col * num_k_blocks];

        for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
            const block_q4_1* w_block = &w_col[kb];
            const float d_w = read_half_as_float(w_block->d);
            const float m_w = read_half_as_float(w_block->m);

            const int k_start = kb * QK4_1;

            float a_block[32];
            const float* act_ptr = &act_row[k_start];

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
                a_block[i * 4 + 0] = a4.x;
                a_block[i * 4 + 1] = a4.y;
                a_block[i * 4 + 2] = a4.z;
                a_block[i * 4 + 3] = a4.w;
            }

            float a_max = 0.0f;
            float a_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                a_max = fmaxf(a_max, fabsf(a_block[i]));
                a_sum += a_block[i];
            }

            const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

            int32_t a_packed[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);

                a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                              ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
            }

            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[i * 4]);

                int8_t w0 = (int8_t)(w_packed & 0x0F);
                int8_t w1 = (int8_t)((w_packed >> 8) & 0x0F);
                int8_t w2 = (int8_t)((w_packed >> 16) & 0x0F);
                int8_t w3 = (int8_t)((w_packed >> 24) & 0x0F);
                int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                            ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);

                w0 = (int8_t)((w_packed >> 4) & 0x0F);
                w1 = (int8_t)((w_packed >> 12) & 0x0F);
                w2 = (int8_t)((w_packed >> 20) & 0x0F);
                w3 = (int8_t)((w_packed >> 28) & 0x0F);
                w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                        ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sum += d_w * d_a * (float)sumi + m_w * a_sum;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            output[row * N + col] = sum;
        }
    }
}

/**
 * Tiled kernel for larger batch sizes
 */
#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define THREADS_M 8
#define THREADS_N 32

__global__ void __launch_bounds__(THREADS_M * THREADS_N)
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

    __shared__ float smem_activation[TILE_M][TILE_K];
    __shared__ int8_t smem_a_quantized[TILE_M][TILE_K];
    __shared__ float smem_a_scale[TILE_M];
    __shared__ float smem_a_sum[TILE_M];
    __shared__ block_q4_1 smem_weight[TILE_N];

    const int items_per_thread_m = TILE_M / THREADS_M;
    const int items_per_thread_n = TILE_N / THREADS_N;

    float accum[8][4];
    #pragma unroll
    for (int i = 0; i < items_per_thread_m; ++i) {
        #pragma unroll
        for (int j = 0; j < items_per_thread_n; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_blocks = K / QK4_1;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * QK4_1;

        for (int m_local = 0; m_local < TILE_M; ++m_local) {
            const int m_global = block_m * TILE_M + m_local;
            for (int k_local = thread_n; k_local < TILE_K; k_local += THREADS_N) {
                const int k_global = k_start + k_local;
                if (m_global < M && k_global < K) {
                    smem_activation[m_local][k_local] = activation[m_global * K + k_global];
                } else {
                    smem_activation[m_local][k_local] = 0.0f;
                }
            }
        }

        __syncthreads();

        if (thread_m == 0) {
            for (int m_local = 0; m_local < TILE_M; ++m_local) {
                float local_max = 0.0f;
                float local_sum = 0.0f;

                for (int k = thread_n; k < TILE_K; k += THREADS_N) {
                    const float val = smem_activation[m_local][k];
                    local_max = fmaxf(local_max, fabsf(val));
                    local_sum += val;
                }

                #pragma unroll
                for (int offset = THREADS_N / 2; offset > 0; offset /= 2) {
                    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
                    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
                }

                if (thread_n == 0) {
                    const float d_a = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
                    smem_a_scale[m_local] = d_a;
                    smem_a_sum[m_local] = local_sum;
                }
            }
        }

        __syncthreads();

        for (int m_local = 0; m_local < TILE_M; ++m_local) {
            for (int k_local = thread_n; k_local < TILE_K; k_local += THREADS_N) {
                const float val = smem_activation[m_local][k_local];
                const float d_a = smem_a_scale[m_local];
                smem_a_quantized[m_local][k_local] = (int8_t)__float2int_rn(val / d_a);
            }
        }

        for (int n_local = tid; n_local < TILE_N; n_local += (THREADS_M * THREADS_N)) {
            const int n_global = block_n * TILE_N + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[n_global * num_k_blocks + k_block];
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < items_per_thread_m; ++i) {
            const int m_local = thread_m * items_per_thread_m + i;
            const int m_global = block_m * TILE_M + m_local;

            if (m_global >= M) continue;

            const float d_a = smem_a_scale[m_local];
            const float s_a = smem_a_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_per_thread_n; ++j) {
                const int n_local = thread_n * items_per_thread_n + j;
                const int n_global = block_n * TILE_N + n_local;

                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                const float d_w = read_half_as_float(w_block->d);
                const float m_w = read_half_as_float(w_block->m);

                int32_t sumi = 0;

                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int a_pack = *reinterpret_cast<const int*>(&smem_a_quantized[m_local][ii * 4]);

                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);
                    const int8_t w0 = (int8_t)(w_packed & 0x0F);
                    const int8_t w1 = (int8_t)((w_packed >> 8) & 0x0F);
                    const int8_t w2 = (int8_t)((w_packed >> 16) & 0x0F);
                    const int8_t w3 = (int8_t)((w_packed >> 24) & 0x0F);
                    const int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                                      ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int a_pack = *reinterpret_cast<const int*>(&smem_a_quantized[m_local][16 + ii * 4]);

                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);
                    const int8_t w0 = (int8_t)((w_packed >> 4) & 0x0F);
                    const int8_t w1 = (int8_t)((w_packed >> 12) & 0x0F);
                    const int8_t w2 = (int8_t)((w_packed >> 20) & 0x0F);
                    const int8_t w3 = (int8_t)((w_packed >> 28) & 0x0F);
                    const int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                                      ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                accum[i][j] += d_w * d_a * (float)sumi + m_w * s_a;
            }
        }

        __syncthreads();
    }

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

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const int BATCH_THRESHOLD = 16;

    if (M <= BATCH_THRESHOLD) {
        const int total_outputs = M * N;
        const int threads_per_block = 256;
        const int warps_per_block = threads_per_block / WARP_SIZE;

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

        const int min_blocks = num_sms * 2;
        const int needed_blocks = (total_outputs + warps_per_block - 1) / warps_per_block;
        const int num_blocks = (needed_blocks > min_blocks) ? needed_blocks : min_blocks;

        gemm_q4_1_warp_kernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
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
    m.def("forward", &forward, "Q4_1 x Q8_1 GEMM - v12");
}
