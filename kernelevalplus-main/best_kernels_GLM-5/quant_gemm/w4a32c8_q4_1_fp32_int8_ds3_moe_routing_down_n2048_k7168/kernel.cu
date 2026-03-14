/**
 * Quantized GEMM for DeepSeek-V3 MoE Routing Down Projection with Q4_1 Weights
 * Version 17: Optimized with __ldg for L2 cache hints
 *
 * Key optimizations:
 * - Warp-level K-parallelism for small batches
 * - __ldg for weight reads to utilize L2 read-only cache
 * - Fully unrolled computation for maximum ILP
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cstdint>

constexpr int QK4_1 = 32;
constexpr int WARP_SIZE = 32;
constexpr int SMALL_BATCH_THRESHOLD = 32;

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

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a_device(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
__device__ __forceinline__ int dp4a_device(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

// ============================================================================
// Kernel 1: Small Batch - Original working version with minor optimizations
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_1_small_batch(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    const int num_blocks = K / QK4_1;

    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int row = idx / N;
        const int col = idx % N;

        float sum = 0.0f;
        const float* act_row = activation + row * K;
        const block_q4_1* w_row = weight + col * num_blocks;

        for (int b = lane_id; b < num_blocks; b += WARP_SIZE) {
            const block_q4_1* w_block = &w_row[b];

            // Use __ldg for L2 cache hint on weights
            const float d_w = half_to_float(__ldg(&w_block->d));
            const float m_w = half_to_float(__ldg(&w_block->m));
            const int k_start = b * QK4_1;

            // Vectorized load of activation
            float a0, a1, a2, a3, a4, a5, a6, a7;
            {
                const float4 a4_0 = *reinterpret_cast<const float4*>(&act_row[k_start]);
                const float4 a4_1 = *reinterpret_cast<const float4*>(&act_row[k_start + 4]);
                a0 = a4_0.x; a1 = a4_0.y; a2 = a4_0.z; a3 = a4_0.w;
                a4 = a4_1.x; a5 = a4_1.y; a6 = a4_1.z; a7 = a4_1.w;
            }
            float a8, a9, a10, a11, a12, a13, a14, a15;
            {
                const float4 a4_2 = *reinterpret_cast<const float4*>(&act_row[k_start + 8]);
                const float4 a4_3 = *reinterpret_cast<const float4*>(&act_row[k_start + 12]);
                a8 = a4_2.x; a9 = a4_2.y; a10 = a4_2.z; a11 = a4_2.w;
                a12 = a4_3.x; a13 = a4_3.y; a14 = a4_3.z; a15 = a4_3.w;
            }
            float a16, a17, a18, a19, a20, a21, a22, a23;
            {
                const float4 a4_4 = *reinterpret_cast<const float4*>(&act_row[k_start + 16]);
                const float4 a4_5 = *reinterpret_cast<const float4*>(&act_row[k_start + 20]);
                a16 = a4_4.x; a17 = a4_4.y; a18 = a4_4.z; a19 = a4_4.w;
                a20 = a4_5.x; a21 = a4_5.y; a22 = a4_5.z; a23 = a4_5.w;
            }
            float a24, a25, a26, a27, a28, a29, a30, a31;
            {
                const float4 a4_6 = *reinterpret_cast<const float4*>(&act_row[k_start + 24]);
                const float4 a4_7 = *reinterpret_cast<const float4*>(&act_row[k_start + 28]);
                a24 = a4_6.x; a25 = a4_6.y; a26 = a4_6.z; a27 = a4_6.w;
                a28 = a4_7.x; a29 = a4_7.y; a30 = a4_7.z; a31 = a4_7.w;
            }

            // Load and unpack weights with __ldg for L2 cache
            const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[0]));
            const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[4]));
            const uint32_t w2 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[8]));
            const uint32_t w3 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[12]));

            // Process weights - fully unrolled
            sum += (d_w * (w0 & 0xF) + m_w) * a0;
            sum += (d_w * ((w0 >> 4) & 0xF) + m_w) * a16;
            sum += (d_w * ((w0 >> 8) & 0xF) + m_w) * a1;
            sum += (d_w * ((w0 >> 12) & 0xF) + m_w) * a17;
            sum += (d_w * ((w0 >> 16) & 0xF) + m_w) * a2;
            sum += (d_w * ((w0 >> 20) & 0xF) + m_w) * a18;
            sum += (d_w * ((w0 >> 24) & 0xF) + m_w) * a3;
            sum += (d_w * ((w0 >> 28) & 0xF) + m_w) * a19;

            sum += (d_w * (w1 & 0xF) + m_w) * a4;
            sum += (d_w * ((w1 >> 4) & 0xF) + m_w) * a20;
            sum += (d_w * ((w1 >> 8) & 0xF) + m_w) * a5;
            sum += (d_w * ((w1 >> 12) & 0xF) + m_w) * a21;
            sum += (d_w * ((w1 >> 16) & 0xF) + m_w) * a6;
            sum += (d_w * ((w1 >> 20) & 0xF) + m_w) * a22;
            sum += (d_w * ((w1 >> 24) & 0xF) + m_w) * a7;
            sum += (d_w * ((w1 >> 28) & 0xF) + m_w) * a23;

            sum += (d_w * (w2 & 0xF) + m_w) * a8;
            sum += (d_w * ((w2 >> 4) & 0xF) + m_w) * a24;
            sum += (d_w * ((w2 >> 8) & 0xF) + m_w) * a9;
            sum += (d_w * ((w2 >> 12) & 0xF) + m_w) * a25;
            sum += (d_w * ((w2 >> 16) & 0xF) + m_w) * a10;
            sum += (d_w * ((w2 >> 20) & 0xF) + m_w) * a26;
            sum += (d_w * ((w2 >> 24) & 0xF) + m_w) * a11;
            sum += (d_w * ((w2 >> 28) & 0xF) + m_w) * a27;

            sum += (d_w * (w3 & 0xF) + m_w) * a12;
            sum += (d_w * ((w3 >> 4) & 0xF) + m_w) * a28;
            sum += (d_w * ((w3 >> 8) & 0xF) + m_w) * a13;
            sum += (d_w * ((w3 >> 12) & 0xF) + m_w) * a29;
            sum += (d_w * ((w3 >> 16) & 0xF) + m_w) * a14;
            sum += (d_w * ((w3 >> 20) & 0xF) + m_w) * a30;
            sum += (d_w * ((w3 >> 24) & 0xF) + m_w) * a15;
            sum += (d_w * ((w3 >> 28) & 0xF) + m_w) * a31;
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            output[row * N + col] = sum;
        }
    }
}

// ============================================================================
// Kernel 2: Large Batch - DP4A-optimized
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

    const int num_k_blocks = K / QK4_1;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * QK4_1;

        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_M * THREADS_N) {
            const int m_local = idx / TILE_K;
            const int k_local = idx % TILE_K;
            const int m_global = block_m * TILE_M + m_local;
            const int k_global = k_start + k_local;
            smem_act[m_local][k_local] = (m_global < M) ? activation[m_global * K + k_global] : 0.0f;
        }
        __syncthreads();

        for (int m_base = 0; m_base < TILE_M; m_base += THREADS_M) {
            const int m_local = m_base + thread_m;
            if (m_local >= TILE_M) continue;

            float local_max = 0.0f, local_sum = 0.0f;
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
                smem_act_scale[m_local] = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
                smem_act_sum[m_local] = local_sum;
            }
        }
        __syncthreads();

        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_M * THREADS_N) {
            const int m_local = idx / TILE_K;
            const int k_local = idx % TILE_K;
            smem_act_q[m_local][k_local] = (int8_t)__float2int_rn(smem_act[m_local][k_local] / smem_act_scale[m_local]);
        }

        for (int n_local = tid; n_local < TILE_N; n_local += THREADS_M * THREADS_N) {
            const int n_global = block_n * TILE_N + n_local;
            if (n_global < N) smem_weight[n_local] = weight[n_global * num_k_blocks + k_block];
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
                    const int w_pack = (w_packed & 0xF) | (((w_packed >> 8) & 0xF) << 8) | (((w_packed >> 16) & 0xF) << 16) | (((w_packed >> 24) & 0xF) << 24);
                    sumi = dp4a_device(a_pack, w_pack, sumi);
                }
                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int a_pack = *reinterpret_cast<const int*>(&smem_act_q[m_local][16 + ii * 4]);
                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);
                    const int w_pack = ((w_packed >> 4) & 0xF) | (((w_packed >> 12) & 0xF) << 8) | (((w_packed >> 20) & 0xF) << 16) | (((w_packed >> 28) & 0xF) << 24);
                    sumi = dp4a_device(a_pack, w_pack, sumi);
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
            if (n_global < N) output[m_global * N + n_global] = accum[i][j];
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
        const int total_outputs = M * N;
        const int threads = 256;
        const int warps_per_block = threads / WARP_SIZE;
        const int num_blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

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
    m.def("forward", &forward, "Q4_1 GEMM v17 - Optimized with __ldg");
}
