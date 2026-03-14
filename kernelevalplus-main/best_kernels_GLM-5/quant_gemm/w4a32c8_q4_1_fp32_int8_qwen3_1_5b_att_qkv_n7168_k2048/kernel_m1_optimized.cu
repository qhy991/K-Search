/**
 * Quantized GEMM for Qwen3-1.5B Attention QKV Projection with Q4_1 Weights
 * v12: Performance optimized - keep warp-based approach, optimize grid config
 *
 * Key optimizations:
 * 1. Use more thread blocks to saturate all 128 SMs
 * 2. Grid-stride loop for better load balancing
 * 3. Keep warp-based K reduction (proven effective in v10)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int SMALL_BATCH_THRESHOLD = 32;
constexpr int NUM_SM = 128;  // RTX 4090

struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size must be 20 bytes");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
#else
    const int8_t* ab = reinterpret_cast<const int8_t*>(&a);
    const int8_t* bb = reinterpret_cast<const int8_t*>(&b);
    return c + ab[0]*bb[0] + ab[1]*bb[1] + ab[2]*bb[2] + ab[3]*bb[3];
#endif
}

// ============================================================================
// Kernel 1: Small Batch - Optimized grid configuration
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_1_small_batch(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    const int num_blocks = K / QK;

    // Grid-stride loop for better load balancing
    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int row = idx / N;
        const int col = idx % N;

        float sum = 0.0f;
        const float* act_row = activation + row * K;

        for (int b = lane_id; b < num_blocks; b += WARP_SIZE) {
            const block_q4_1* w_block = &weight[col * num_blocks + b];

            const float d_w = half_to_float(w_block->d);
            const float m_w = half_to_float(w_block->m);

            const int k_start = b * QK;

            float a_local[32];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 a4 = *reinterpret_cast<const float4*>(&act_row[k_start + i * 4]);
                a_local[i*4+0] = a4.x;
                a_local[i*4+1] = a4.y;
                a_local[i*4+2] = a4.z;
                a_local[i*4+3] = a4.w;
            }

            float block_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t packed = w_block->qs[i];
                const int w0 = packed & 0x0F;
                const int w1 = (packed >> 4) & 0x0F;

                const float w_deq0 = d_w * w0 + m_w;
                const float w_deq1 = d_w * w1 + m_w;

                block_sum += a_local[i] * w_deq0;
                block_sum += a_local[i + 16] * w_deq1;
            }
            sum += block_sum;
        }

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
// Kernel 2: Large Batch - Tiled DP4A
// ============================================================================
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;
constexpr int THREADS_M = 8;
constexpr int THREADS_N = 32;

__global__ void __launch_bounds__(256) gemm_q4_1_large_batch(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;

    const int thread_m = threadIdx.y;
    const int thread_n = threadIdx.x;
    const int tid = thread_m * THREADS_N + thread_n;

    __shared__ float smem_act[TILE_M][TILE_K];
    __shared__ int8_t smem_act_q[TILE_M][TILE_K];
    __shared__ float smem_act_scale[TILE_M];
    __shared__ float smem_act_sum[TILE_M];
    __shared__ block_q4_1 smem_weight[TILE_N];

    const int items_per_thread_m = TILE_M / THREADS_M;
    const int items_per_thread_n = TILE_N / THREADS_N;

    float accum[items_per_thread_m][items_per_thread_n];
    #pragma unroll
    for (int i = 0; i < items_per_thread_m; ++i) {
        #pragma unroll
        for (int j = 0; j < items_per_thread_n; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_blocks = K / QK;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * QK;

        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_M * THREADS_N) {
            const int m_local = idx / TILE_K;
            const int k_local = idx % TILE_K;
            const int m_global = block_m * TILE_M + m_local;
            const int k_global = k_start + k_local;

            if (m_global < M) {
                smem_act[m_local][k_local] = activation[m_global * K + k_global];
            } else {
                smem_act[m_local][k_local] = 0.0f;
            }
        }
        __syncthreads();

        for (int m_base = 0; m_base < TILE_M; m_base += THREADS_M) {
            const int m_local = m_base + thread_m;
            if (m_local >= TILE_M) continue;

            float local_max = 0.0f;
            float local_sum = 0.0f;

            for (int k = thread_n; k < TILE_K; k += THREADS_N) {
                const float val = smem_act[m_local][k];
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
                smem_act_scale[m_local] = d_a;
                smem_act_sum[m_local] = local_sum;
            }
        }
        __syncthreads();

        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_M * THREADS_N) {
            const int m_local = idx / TILE_K;
            const int k_local = idx % TILE_K;
            const float val = smem_act[m_local][k_local];
            const float d_a = smem_act_scale[m_local];
            smem_act_q[m_local][k_local] = (int8_t)__float2int_rn(val / d_a);
        }

        for (int n_local = tid; n_local < TILE_N; n_local += THREADS_M * THREADS_N) {
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

            const float d_a = smem_act_scale[m_local];
            const float s_a = smem_act_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_per_thread_n; ++j) {
                const int n_local = thread_n * items_per_thread_n + j;
                const int n_global = block_n * TILE_N + n_local;
                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                const float d_w = half_to_float(w_block->d);
                const float m_w = half_to_float(w_block->m);

                int32_t sumi = 0;

                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int a_pack = *reinterpret_cast<const int*>(&smem_act_q[m_local][ii * 4]);
                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);

                    const int w0 = w_packed & 0x0F;
                    const int w1 = (w_packed >> 8) & 0x0F;
                    const int w2 = (w_packed >> 16) & 0x0F;
                    const int w3 = (w_packed >> 24) & 0x0F;

                    const int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) |
                                       ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int a_pack = *reinterpret_cast<const int*>(&smem_act_q[m_local][16 + ii * 4]);
                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);

                    const int w0 = (w_packed >> 4) & 0x0F;
                    const int w1 = (w_packed >> 12) & 0x0F;
                    const int w2 = (w_packed >> 20) & 0x0F;
                    const int w3 = (w_packed >> 28) & 0x0F;

                    const int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) |
                                       ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);

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

// ============================================================================
// PyTorch Interface
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

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const block_q4_1* weight_ptr = reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>());

    if (M <= SMALL_BATCH_THRESHOLD) {
        // Small batch: use warp-based kernel
        // Launch enough blocks to saturate all SMs
        const int total_outputs = M * N;
        const int threads = 256;
        const int warps_per_block = threads / WARP_SIZE;  // 8 warps per block

        // Calculate blocks needed: at least NUM_SM blocks, but not more than needed
        const int min_blocks = NUM_SM;  // At least one block per SM
        const int blocks_needed = (total_outputs + warps_per_block - 1) / warps_per_block;
        const int num_blocks = max(min_blocks, min(blocks_needed, NUM_SM * 4));

        gemm_q4_1_small_batch<<<num_blocks, threads>>>(
            weight_ptr,
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

        gemm_q4_1_large_batch<<<blocks, threads>>>(
            weight_ptr,
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM for Qwen3-1.5B QKV");
}
