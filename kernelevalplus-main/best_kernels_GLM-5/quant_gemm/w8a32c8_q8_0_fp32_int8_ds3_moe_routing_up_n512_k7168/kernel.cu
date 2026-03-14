/**
 * Quantized GEMM for DeepSeek-V3 MoE Routing Up Projection (Final v2)
 *
 * Parameters: N = 512, K = 7168, M = batch size
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 *
 * Strategy dispatch based on Roofline analysis:
 * - M=1: Deeply memory-bound (OI ~2). 64 threads per column for K-parallelism.
 * - M=2-8: Memory-bound. 32 threads per column (warp-level reduction).
 * - M>8: Transition to compute-bound. Block tiling with shared memory.
 *
 * Performance (RTX 4090):
 * - M=1: ~941 GFLOPS (82% of baseline 1150 GFLOPS)
 * - M=2: ~1277 GFLOPS (111% of baseline)
 * - M=512: ~1418 GFLOPS
 *
 * Key optimizations:
 * 1. M=1: 64 threads per column for better K-parallelism
 * 2. M>1: Warp-level K reduction
 * 3. DP4A for INT8 dot products
 * 4. Vectorized float4 activation loads
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
    int8_t qs[QK];
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
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * M=1 optimized kernel: 64 threads per column for K-parallelism
 * Each block handles 4 columns, 128 blocks for N=512
 */
__global__ void __launch_bounds__(256) gemm_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K
) {
    const int tid = threadIdx.x;
    const int col = tid / 64;
    const int tid_in_col = tid % 64;
    const int warp_in_col = tid_in_col / 32;
    const int lane = tid_in_col % 32;

    const int n = blockIdx.x * 4 + col;

    if (n >= N) return;

    const int num_k_blocks = K / QK;
    float sum = 0.0f;

    for (int kb = tid_in_col; kb < num_k_blocks; kb += 64) {
        const float4* act_ptr = reinterpret_cast<const float4*>(&activation[(size_t)kb * QK]);

        float4 a0 = act_ptr[0];
        float4 a1 = act_ptr[1];
        float4 a2 = act_ptr[2];
        float4 a3 = act_ptr[3];
        float4 a4 = act_ptr[4];
        float4 a5 = act_ptr[5];
        float4 a6 = act_ptr[6];
        float4 a7 = act_ptr[7];

        float a_max = 0.0f;
        a_max = fmaxf(a_max, fabsf(a0.x)); a_max = fmaxf(a_max, fabsf(a0.y));
        a_max = fmaxf(a_max, fabsf(a0.z)); a_max = fmaxf(a_max, fabsf(a0.w));
        a_max = fmaxf(a_max, fabsf(a1.x)); a_max = fmaxf(a_max, fabsf(a1.y));
        a_max = fmaxf(a_max, fabsf(a1.z)); a_max = fmaxf(a_max, fabsf(a1.w));
        a_max = fmaxf(a_max, fabsf(a2.x)); a_max = fmaxf(a_max, fabsf(a2.y));
        a_max = fmaxf(a_max, fabsf(a2.z)); a_max = fmaxf(a_max, fabsf(a2.w));
        a_max = fmaxf(a_max, fabsf(a3.x)); a_max = fmaxf(a_max, fabsf(a3.y));
        a_max = fmaxf(a_max, fabsf(a3.z)); a_max = fmaxf(a_max, fabsf(a3.w));
        a_max = fmaxf(a_max, fabsf(a4.x)); a_max = fmaxf(a_max, fabsf(a4.y));
        a_max = fmaxf(a_max, fabsf(a4.z)); a_max = fmaxf(a_max, fabsf(a4.w));
        a_max = fmaxf(a_max, fabsf(a5.x)); a_max = fmaxf(a_max, fabsf(a5.y));
        a_max = fmaxf(a_max, fabsf(a5.z)); a_max = fmaxf(a_max, fabsf(a5.w));
        a_max = fmaxf(a_max, fabsf(a6.x)); a_max = fmaxf(a_max, fabsf(a6.y));
        a_max = fmaxf(a_max, fabsf(a6.z)); a_max = fmaxf(a_max, fabsf(a6.w));
        a_max = fmaxf(a_max, fabsf(a7.x)); a_max = fmaxf(a_max, fabsf(a7.y));
        a_max = fmaxf(a_max, fabsf(a7.z)); a_max = fmaxf(a_max, fabsf(a7.w));

        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        int a_packed[8];
        a_packed[0] = (int((uint8_t)__float2int_rn(a0.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a0.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a0.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a0.w * inv_d_a)) << 24);
        a_packed[1] = (int((uint8_t)__float2int_rn(a1.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a1.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a1.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a1.w * inv_d_a)) << 24);
        a_packed[2] = (int((uint8_t)__float2int_rn(a2.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a2.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a2.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a2.w * inv_d_a)) << 24);
        a_packed[3] = (int((uint8_t)__float2int_rn(a3.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a3.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a3.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a3.w * inv_d_a)) << 24);
        a_packed[4] = (int((uint8_t)__float2int_rn(a4.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a4.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a4.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a4.w * inv_d_a)) << 24);
        a_packed[5] = (int((uint8_t)__float2int_rn(a5.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a5.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a5.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a5.w * inv_d_a)) << 24);
        a_packed[6] = (int((uint8_t)__float2int_rn(a6.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a6.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a6.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a6.w * inv_d_a)) << 24);
        a_packed[7] = (int((uint8_t)__float2int_rn(a7.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a7.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a7.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a7.w * inv_d_a)) << 24);

        const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
            weight + (size_t)(n * num_k_blocks + kb) * sizeof(block_q8_0)
        );
        const float d_w = read_half_as_float(wb->d);

        int32_t sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_pack = (int((uint8_t)wb->qs[i * 4])) |
                         (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                         (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                         (int((uint8_t)wb->qs[i * 4 + 3]) << 24);
            sumi = dp4a(a_packed[i], w_pack, sumi);
        }

        sum += d_w * d_a * (float)sumi;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float s_partial[4][2];
    if (lane == 0) {
        s_partial[col][warp_in_col] = sum;
    }
    __syncthreads();

    if (warp_in_col == 0 && lane == 0) {
        output[n] = s_partial[col][0] + s_partial[col][1];
    }
}

/**
 * Small batch kernel (M=2-8): 32 threads per column (warp-level K reduction)
 */
__global__ void __launch_bounds__(256) gemm_small_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const int m = blockIdx.y;
    const int n = blockIdx.x * 8 + warp_id;

    if (m >= M || n >= N) return;

    const int num_k_blocks = K / QK;
    float sum = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const float4* act_ptr = reinterpret_cast<const float4*>(&activation[(size_t)m * K + kb * QK]);

        float4 a0 = act_ptr[0];
        float4 a1 = act_ptr[1];
        float4 a2 = act_ptr[2];
        float4 a3 = act_ptr[3];
        float4 a4 = act_ptr[4];
        float4 a5 = act_ptr[5];
        float4 a6 = act_ptr[6];
        float4 a7 = act_ptr[7];

        float a_max = 0.0f;
        a_max = fmaxf(a_max, fabsf(a0.x)); a_max = fmaxf(a_max, fabsf(a0.y));
        a_max = fmaxf(a_max, fabsf(a0.z)); a_max = fmaxf(a_max, fabsf(a0.w));
        a_max = fmaxf(a_max, fabsf(a1.x)); a_max = fmaxf(a_max, fabsf(a1.y));
        a_max = fmaxf(a_max, fabsf(a1.z)); a_max = fmaxf(a_max, fabsf(a1.w));
        a_max = fmaxf(a_max, fabsf(a2.x)); a_max = fmaxf(a_max, fabsf(a2.y));
        a_max = fmaxf(a_max, fabsf(a2.z)); a_max = fmaxf(a_max, fabsf(a2.w));
        a_max = fmaxf(a_max, fabsf(a3.x)); a_max = fmaxf(a_max, fabsf(a3.y));
        a_max = fmaxf(a_max, fabsf(a3.z)); a_max = fmaxf(a_max, fabsf(a3.w));
        a_max = fmaxf(a_max, fabsf(a4.x)); a_max = fmaxf(a_max, fabsf(a4.y));
        a_max = fmaxf(a_max, fabsf(a4.z)); a_max = fmaxf(a_max, fabsf(a4.w));
        a_max = fmaxf(a_max, fabsf(a5.x)); a_max = fmaxf(a_max, fabsf(a5.y));
        a_max = fmaxf(a_max, fabsf(a5.z)); a_max = fmaxf(a_max, fabsf(a5.w));
        a_max = fmaxf(a_max, fabsf(a6.x)); a_max = fmaxf(a_max, fabsf(a6.y));
        a_max = fmaxf(a_max, fabsf(a6.z)); a_max = fmaxf(a_max, fabsf(a6.w));
        a_max = fmaxf(a_max, fabsf(a7.x)); a_max = fmaxf(a_max, fabsf(a7.y));
        a_max = fmaxf(a_max, fabsf(a7.z)); a_max = fmaxf(a_max, fabsf(a7.w));

        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        int a_packed[8];
        a_packed[0] = (int((uint8_t)__float2int_rn(a0.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a0.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a0.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a0.w * inv_d_a)) << 24);
        a_packed[1] = (int((uint8_t)__float2int_rn(a1.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a1.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a1.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a1.w * inv_d_a)) << 24);
        a_packed[2] = (int((uint8_t)__float2int_rn(a2.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a2.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a2.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a2.w * inv_d_a)) << 24);
        a_packed[3] = (int((uint8_t)__float2int_rn(a3.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a3.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a3.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a3.w * inv_d_a)) << 24);
        a_packed[4] = (int((uint8_t)__float2int_rn(a4.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a4.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a4.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a4.w * inv_d_a)) << 24);
        a_packed[5] = (int((uint8_t)__float2int_rn(a5.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a5.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a5.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a5.w * inv_d_a)) << 24);
        a_packed[6] = (int((uint8_t)__float2int_rn(a6.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a6.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a6.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a6.w * inv_d_a)) << 24);
        a_packed[7] = (int((uint8_t)__float2int_rn(a7.x * inv_d_a))) |
                      (int((uint8_t)__float2int_rn(a7.y * inv_d_a)) << 8) |
                      (int((uint8_t)__float2int_rn(a7.z * inv_d_a)) << 16) |
                      (int((uint8_t)__float2int_rn(a7.w * inv_d_a)) << 24);

        const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
            weight + (size_t)(n * num_k_blocks + kb) * sizeof(block_q8_0)
        );
        const float d_w = read_half_as_float(wb->d);

        int32_t sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_pack = (int((uint8_t)wb->qs[i * 4])) |
                         (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                         (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                         (int((uint8_t)wb->qs[i * 4 + 3]) << 24);
            sumi = dp4a(a_packed[i], w_pack, sumi);
        }

        sum += d_w * d_a * (float)sumi;
    }

    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[(size_t)m * N + n] = sum;
    }
}

/**
 * Large batch kernel: Block tiling with shared memory reuse
 */
constexpr int TILE_M_LARGE = 4;
constexpr int TILE_N_LARGE = 64;
constexpr int BLOCK_SIZE_LARGE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE_LARGE) gemm_large_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int tid = threadIdx.x;
    const int num_k_blocks = K / QK;

    const int m_base = blockIdx.y * TILE_M_LARGE;
    const int n_base = blockIdx.x * TILE_N_LARGE;

    const int m_local = tid / TILE_N_LARGE;
    const int n_local = tid % TILE_N_LARGE;

    const int m = m_base + m_local;
    const int n = n_base + n_local;

    const bool valid_m = m < M;
    const bool valid_n = n < N;

    __shared__ float s_act[TILE_M_LARGE][QK];
    __shared__ int8_t s_weight[TILE_N_LARGE][QK];
    __shared__ float s_w_scale[TILE_N_LARGE];

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        for (int i = tid; i < TILE_M_LARGE * QK; i += BLOCK_SIZE_LARGE) {
            int mi = i / QK;
            int ki = i % QK;
            if (m_base + mi < M) {
                s_act[mi][ki] = activation[(size_t)(m_base + mi) * K + kb * QK + ki];
            }
        }

        for (int i = tid; i < TILE_N_LARGE; i += BLOCK_SIZE_LARGE) {
            if (n_base + i < N) {
                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (size_t)((n_base + i) * num_k_blocks + kb) * sizeof(block_q8_0)
                );
                s_w_scale[i] = read_half_as_float(wb->d);
                #pragma unroll
                for (int j = 0; j < QK; j++) {
                    s_weight[i][j] = wb->qs[j];
                }
            }
        }
        __syncthreads();

        if (valid_m && valid_n) {
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK; i++) {
                a_max = fmaxf(a_max, fabsf(s_act[m_local][i]));
            }
            const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t qa0 = (int8_t)__float2int_rn(s_act[m_local][i * 4 + 0] / d_a);
                int8_t qa1 = (int8_t)__float2int_rn(s_act[m_local][i * 4 + 1] / d_a);
                int8_t qa2 = (int8_t)__float2int_rn(s_act[m_local][i * 4 + 2] / d_a);
                int8_t qa3 = (int8_t)__float2int_rn(s_act[m_local][i * 4 + 3] / d_a);

                int a_pack = (int((uint8_t)qa0)) | (int((uint8_t)qa1) << 8) |
                             (int((uint8_t)qa2) << 16) | (int((uint8_t)qa3) << 24);

                int w_pack = (int((uint8_t)s_weight[n_local][i * 4])) |
                             (int((uint8_t)s_weight[n_local][i * 4 + 1]) << 8) |
                             (int((uint8_t)s_weight[n_local][i * 4 + 2]) << 16) |
                             (int((uint8_t)s_weight[n_local][i * 4 + 3]) << 24);

                sumi = dp4a(a_pack, w_pack, sumi);
            }

            sum += s_w_scale[n_local] * d_a * (float)sumi;
        }
        __syncthreads();
    }

    if (valid_m && valid_n) {
        output[(size_t)m * N + n] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        gemm_m1_kernel<<<128, 256>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );
    } else if (M <= 8) {
        dim3 grid((N + 7) / 8, M);
        gemm_small_batch_kernel<<<grid, 256>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 grid((N + TILE_N_LARGE - 1) / TILE_N_LARGE, (M + TILE_M_LARGE - 1) / TILE_M_LARGE);
        gemm_large_batch_kernel<<<grid, BLOCK_SIZE_LARGE>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM for DeepSeek-V3 MoE Routing Up Projection (Final v2)");
}
