/**
 * Quantized GEMM for DeepSeek-V2 MoE Down Projection with Q8_0 Weights - Final
 *
 * Parameters:
 *   - N = 5120, K = 12288, M = batch size
 *
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 * Formula: result = d_w * d_a * sumi
 *
 * Strategy Dispatch:
 *   - M <= 4:  Warp-parallel kernel (maximize parallelism for small batch)
 *   - M <= 16: Optimized small-M kernel with better occupancy
 *   - M > 16:  2D tiled kernel for throughput
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

typedef struct {
    uint16_t d;
    int8_t qs[32];
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size must be 34 bytes");

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void load_weight_block(
    const uint8_t* weight, int n, int b, int num_blocks_k,
    float& d_w, int w_qs_packed[8]
) {
    const size_t offset = (size_t)n * num_blocks_k * 34 + (size_t)b * 34;
    d_w = read_half_as_float(weight[offset] | (weight[offset + 1] << 8));
    for (int i = 0; i < 8; i++) {
        int idx = offset + 2 + i * 4;
        w_qs_packed[i] = (int)(uint8_t)weight[idx] |
                         ((int)(uint8_t)weight[idx + 1] << 8) |
                         ((int)(uint8_t)weight[idx + 2] << 16) |
                         ((int)(uint8_t)weight[idx + 3] << 24);
    }
}

// Strategy 1: Warp-parallel for M=1-4
__global__ void __launch_bounds__(128) gemm_warp_parallel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = (gridDim.x * blockDim.x) >> 5;
    const int num_blocks_k = K / QK;

    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int m = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
            float d_w;
            int w_qs_packed[8];
            load_weight_block(weight, n, b, num_blocks_k, d_w, w_qs_packed);

            const int k_start = b * QK;
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK; i++) {
                a_max = fmaxf(a_max, fabsf(activation[m * K + k_start + i]));
            }
            const float d_a = a_max > 0 ? a_max / 127.0f : 1.0f;

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t a0 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4] / d_a);
                int8_t a1 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4 + 1] / d_a);
                int8_t a2 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4 + 2] / d_a);
                int8_t a3 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4 + 3] / d_a);
                int a_pack = (int((uint8_t)a0)) | (int((uint8_t)a1) << 8) |
                             (int((uint8_t)a2) << 16) | (int((uint8_t)a3) << 24);
                sumi = dp4a(a_pack, w_qs_packed[i], sumi);
            }
            sum += d_w * d_a * (float)sumi;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            output[m * N + n] = sum;
        }
    }
}

// Strategy 2: Thread-coarsened for M=5-16
__global__ void __launch_bounds__(256) gemm_thread_coarsened(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = gridDim.x * blockDim.x;
    const int num_blocks_k = K / QK;
    const int N_PER_THREAD = 4;

    for (int base_idx = tid * N_PER_THREAD; base_idx < M * N; base_idx += num_threads * N_PER_THREAD) {
        float sums[N_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; i++) sums[i] = 0.0f;

        const int m = base_idx / N;
        const int n_start = base_idx % N;

        for (int b = 0; b < num_blocks_k; b++) {
            const int k_start = b * QK;

            // Quantize activation once per K-block
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK; i++) {
                a_max = fmaxf(a_max, fabsf(activation[m * K + k_start + i]));
            }
            const float d_a = a_max > 0 ? a_max / 127.0f : 1.0f;

            int32_t a_qs[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t q0 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(activation[m * K + k_start + i*4 + 3] / d_a);
                a_qs[i] = (int((uint8_t)q0)) | (int((uint8_t)q1) << 8) |
                          (int((uint8_t)q2) << 16) | (int((uint8_t)q3) << 24);
            }

            // Process N outputs
            #pragma unroll
            for (int n_offset = 0; n_offset < N_PER_THREAD; n_offset++) {
                const int n = n_start + n_offset;
                if (n >= N) continue;

                float d_w;
                int w_qs_packed[8];
                load_weight_block(weight, n, b, num_blocks_k, d_w, w_qs_packed);

                int32_t sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    sumi = dp4a(a_qs[i], w_qs_packed[i], sumi);
                }
                sums[n_offset] += d_w * d_a * (float)sumi;
            }
        }

        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; i++) {
            const int n = n_start + i;
            if (n < N) {
                output[m * N + n] = sums[i];
            }
        }
    }
}

// Strategy 3: 2D Tiled for M > 16
__global__ void __launch_bounds__(256) gemm_2d_tiled(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int num_blocks_k = K / QK;

    const int TM = 4, TN = 64;
    const int m_base = blockIdx.y * TM;
    const int n_base = blockIdx.x * TN;

    float acc[TM][TN / 8];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 8; j++)
            acc[i][j] = 0.0f;

    __shared__ float smem_w_d[TN];
    __shared__ int32_t smem_w_qs[TN][8];

    for (int b = 0; b < num_blocks_k; b++) {
        const int k_start = b * QK;

        // Load weights
        for (int n_local = tid; n_local < TN; n_local += 256) {
            const int n_global = n_base + n_local;
            if (n_global < N) {
                float d_w;
                int w_qs_packed[8];
                load_weight_block(weight, n_global, b, num_blocks_k, d_w, w_qs_packed);
                smem_w_d[n_local] = d_w;
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    smem_w_qs[n_local][i] = w_qs_packed[i];
            }
        }
        __syncthreads();

        // Compute
        for (int m_local = 0; m_local < TM; m_local++) {
            const int m_global = m_base + m_local;
            if (m_global >= M) continue;

            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK; i++)
                a_max = fmaxf(a_max, fabsf(activation[m_global * K + k_start + i]));
            const float d_a = a_max > 0 ? a_max / 127.0f : 1.0f;

            int32_t a_qs[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t q0 = (int8_t)__float2int_rn(activation[m_global * K + k_start + i*4] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(activation[m_global * K + k_start + i*4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(activation[m_global * K + k_start + i*4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(activation[m_global * K + k_start + i*4 + 3] / d_a);
                a_qs[i] = (int((uint8_t)q0)) | (int((uint8_t)q1) << 8) |
                          (int((uint8_t)q2) << 16) | (int((uint8_t)q3) << 24);
            }

            const int n_warp_base = warp_id * (TN / 8);
            #pragma unroll
            for (int n_offset = 0; n_offset < TN / 8; n_offset++) {
                const int n_local = n_warp_base + n_offset;
                const int n_global = n_base + n_local;
                if (n_global >= N) continue;

                int32_t sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    sumi = dp4a(a_qs[j], smem_w_qs[n_local][j], sumi);
                acc[m_local][n_offset] += smem_w_d[n_local] * d_a * (float)sumi;
            }
        }
        __syncthreads();
    }

    // Write
    for (int m_local = 0; m_local < TM; m_local++) {
        const int m_global = m_base + m_local;
        if (m_global >= M) continue;
        const int n_warp_base = warp_id * (TN / 8);
        #pragma unroll
        for (int n_offset = 0; n_offset < TN / 8; n_offset++) {
            const int n_local = n_warp_base + n_offset;
            const int n_global = n_base + n_local;
            if (n_global < N)
                output[m_global * N + n_global] = acc[m_local][n_offset];
        }
    }
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 8) {
        // Strategy 1: Warp-parallel (best for small M)
        const int threads = 128;
        const int blocks = min((M * N + threads / WARP_SIZE - 1) / (threads / WARP_SIZE), 128);
        gemm_warp_parallel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        // Strategy 3: 2D Tiled (best for larger M)
        const int TM = 4, TN = 64;
        dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
        gemm_2d_tiled<<<grid, 256>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM Final");
}
