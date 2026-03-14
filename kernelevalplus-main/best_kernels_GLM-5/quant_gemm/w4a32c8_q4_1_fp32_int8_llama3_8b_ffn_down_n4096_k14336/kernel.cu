/**
 * LLaMA-3-8B FFN Down Projection - Q4_1 Quantized GEMM v7
 *
 * Task: C = A @ W^T where A is M×K FP32, W is N×(K/32) Q4_1 quantized
 * Dimensions: N=4096, K=14336, M varies (1-512)
 *
 * FINAL OPTIMIZED STRATEGY:
 *   - M <= 16: Warp kernel with SM saturation (consistent ~2300+ GFLOPS)
 *   - M > 16:  Tiled kernel with shared memory (~4750+ GFLOPS)
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
    return __half2float(reinterpret_cast<half&>(h));
}

inline __device__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Strategy 1: Warp kernel for M <= 16
// ============================================================================

__global__ void __launch_bounds__(256)
gemm_warp_kernel(
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
        const float* act_row = &activation[static_cast<int64_t>(row) * K];
        const block_q4_1* w_col = &weight[static_cast<int64_t>(col) * num_k_blocks];

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

        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            output[row * N + col] = sum;
        }
    }
}

// ============================================================================
// Strategy 2: Tiled kernel for M > 16
// ============================================================================

constexpr int TILE_M = 32;
constexpr int TILE_N = 64;
constexpr int THREADS_M = 4;
constexpr int THREADS_N = 32;

__global__ void gemm_tiled_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int tid = threadIdx.y * THREADS_N + threadIdx.x;
    const int thread_m = threadIdx.y;
    const int thread_n = threadIdx.x;

    __shared__ float smem_act[TILE_M][32];
    __shared__ int8_t smem_a_qs[TILE_M][32];
    __shared__ float smem_a_scale[TILE_M];
    __shared__ float smem_a_sum[TILE_M];
    __shared__ block_q4_1 smem_weight[TILE_N];

    const int items_m = TILE_M / THREADS_M;
    const int items_n = TILE_N / THREADS_N;

    float accum[8][2];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_blocks = K / 32;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int k_start = k_block * 32;

        // Load activation tile
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            const int m_global = block_m * TILE_M + m_local;

            if (m_global < M && m_local < TILE_M) {
                const float* act_ptr = &activation[static_cast<int64_t>(m_global) * K + k_start];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[k * 4]);
                    smem_act[m_local][k * 4] = a4.x;
                    smem_act[m_local][k * 4 + 1] = a4.y;
                    smem_act[m_local][k * 4 + 2] = a4.z;
                    smem_act[m_local][k * 4 + 3] = a4.w;
                }
            }
        }

        __syncthreads();

        // Compute Q8_1 statistics
        if (thread_n == 0) {
            #pragma unroll
            for (int i = 0; i < items_m; i++) {
                const int m_local = thread_m * items_m + i;
                if (m_local >= TILE_M) continue;

                float a_max = 0.0f;
                float a_sum = 0.0f;

                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    float val = smem_act[m_local][k];
                    a_max = fmaxf(a_max, fabsf(val));
                    a_sum += val;
                }

                smem_a_scale[m_local] = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
                smem_a_sum[m_local] = a_sum;
            }
        }

        __syncthreads();

        // Quantize activation
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            if (m_local >= TILE_M) continue;

            const float d_a = smem_a_scale[m_local];

            #pragma unroll
            for (int k = 0; k < 32; k++) {
                smem_a_qs[m_local][k] = (int8_t)__float2int_rn(smem_act[m_local][k] / d_a);
            }
        }

        // Load weights
        for (int n_local = tid; n_local < TILE_N; n_local += (THREADS_M * THREADS_N)) {
            const int n_global = block_n * TILE_N + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[static_cast<int64_t>(n_global) * num_k_blocks + k_block];
            }
        }

        __syncthreads();

        // Compute matrix multiplication
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            const int m_global = block_m * TILE_M + m_local;
            if (m_global >= M || m_local >= TILE_M) continue;

            const float d_a = smem_a_scale[m_local];
            const float s_a = smem_a_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_n; j++) {
                const int n_local = thread_n * items_n + j;
                const int n_global = block_n * TILE_N + n_local;
                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                const float d_w = read_half_as_float(w_block->d);
                const float m_w = read_half_as_float(w_block->m);

                int32_t sumi = 0;

                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_qs[m_local][g * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&w_block->qs[g * 4]);
                    int8_t w0 = (int8_t)(w_raw & 0x0F);
                    int8_t w1 = (int8_t)((w_raw >> 8) & 0x0F);
                    int8_t w2 = (int8_t)((w_raw >> 16) & 0x0F);
                    int8_t w3 = (int8_t)((w_raw >> 24) & 0x0F);
                    int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                                ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_qs[m_local][16 + g * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&w_block->qs[g * 4]);
                    int8_t w0 = (int8_t)((w_raw >> 4) & 0x0F);
                    int8_t w1 = (int8_t)((w_raw >> 12) & 0x0F);
                    int8_t w2 = (int8_t)((w_raw >> 20) & 0x0F);
                    int8_t w3 = (int8_t)((w_raw >> 28) & 0x0F);
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
    for (int i = 0; i < 8; i++) {
        const int m_global = block_m * TILE_M + thread_m * items_m + i;
        if (m_global >= M) continue;

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int n_global = block_n * TILE_N + thread_n * items_n + j;
            if (n_global < N) {
                output[m_global * N + n_global] = accum[i][j];
            }
        }
    }
}

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 16) {
        // M <= 16: Warp kernel with SM saturation
        const int total_outputs = M * N;
        const int threads_per_block = 256;
        const int warps_per_block = threads_per_block / WARP_SIZE;

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

        const int min_blocks = num_sms * 2;
        const int needed_blocks = (total_outputs + warps_per_block - 1) / warps_per_block;
        const int num_blocks = (needed_blocks > min_blocks) ? needed_blocks : min_blocks;

        gemm_warp_kernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // M > 16: Tiled kernel
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks(
            (M + TILE_M - 1) / TILE_M,
            (N + TILE_N - 1) / TILE_N
        );

        gemm_tiled_kernel<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LLaMA-3-8B FFN Down Q4_1 GEMM v7");
}
