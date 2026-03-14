#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v7 - Adapted Best Pattern)
 *
 * Qwen3-4B FFN Down projection
 * - N = 2560 (output features)
 * - K = 9728 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Based on kernel_best.cu pattern, adapted for this problem.
 * Uses DP4A with dynamic Q8_1 quantization for FP32 activation.
 */

#define QK 32
#define Q4_0_BYTES 18

__device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t *va = (const int8_t*)&a;
    const int8_t *vb = (const int8_t*)&b;
    return c + va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2] + va[3]*vb[3];
#endif
}

__device__ __forceinline__ float read_fp16(const uint8_t* p) {
    uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    union { uint16_t u16; __half f16; } c;
    c.u16 = u;
    return __half2float(c.f16);
}

// Dot product with dynamic Q8_1 quantization
__device__ __forceinline__ float dot_q4_0_fp32_block(
    const uint8_t* __restrict__ w_ptr,
    const float* __restrict__ a_ptr
) {
    float d_w = read_fp16(w_ptr);

    // Compute Q8_1 statistics for this activation block
    float amax = 0.0f;
    float asum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        float v = a_ptr[i];
        asum += v;
        amax = fmaxf(amax, fabsf(v));
    }

    float d_a = amax / 127.0f;
    if (d_a < 1e-10f) d_a = 1.0f;
    float inv_d = 1.0f / d_a;

    // Quantize activation to INT8
    int8_t aq[QK];
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        int v = __float2int_rn(a_ptr[i] * inv_d);
        aq[i] = (int8_t)max(-128, min(127, v));
    }

    // Compute integer dot product using DP4A
    int sumi = 0;
    const uint8_t* w_qs = w_ptr + 2;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int8_t tl[4], th[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t b = w_qs[i*4 + j];
            tl[j] = (int8_t)(b & 0x0F);
            th[j] = (int8_t)((b >> 4) & 0x0F);
        }
        sumi = dp4a_i8(*reinterpret_cast<int*>(tl), *reinterpret_cast<int*>(&aq[i*4]), sumi);
        sumi = dp4a_i8(*reinterpret_cast<int*>(th), *reinterpret_cast<int*>(&aq[16+i*4]), sumi);
    }

    // Apply compensation: d_w * (d_a * sumi - 8 * asum)
    return d_w * (d_a * (float)sumi - 8.0f * asum);
}

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        val += __shfl_down_sync(0xffffffff, val, off);
    }
    return val;
}

// Small M kernel
#define K_THREADS_SMALL 4
#define N_PER_WARP (32 / K_THREADS_SMALL)

__global__ void gemm_small_dp4a(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;
    int lane = threadIdx.x;
    int n_local = lane / K_THREADS_SMALL;
    int k_part = lane % K_THREADS_SMALL;
    int n = blockIdx.x * N_PER_WARP + n_local;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    int blocks_per_thread = (num_blocks + K_THREADS_SMALL - 1) / K_THREADS_SMALL;
    int b_start = k_part * blocks_per_thread;
    int b_end = min(b_start + blocks_per_thread, num_blocks);

    float partial = 0.0f;
    for (int b = b_start; b < b_end; b++) {
        partial += dot_q4_0_fp32_block(
            &weight[(n * num_blocks + b) * Q4_0_BYTES],
            &activation[m * K + b * QK]
        );
    }

    partial = warp_reduce_sum(partial);

    if (k_part == 0 && n < N) {
        output[m * N + n] = partial;
    }
}

// Large M kernel with shared memory
#define TILE_M_SM 4
#define TILE_N_SM 64

__global__ void gemm_large_shared(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;

    int tm = threadIdx.y;
    int tn = threadIdx.x;

    int m_base = blockIdx.y * TILE_M_SM + tm;
    int n_base = blockIdx.x * TILE_N_SM + tn;

    if (m_base >= M || n_base >= N) return;

    float acc = 0.0f;

    for (int kb = 0; kb < num_blocks; kb++) {
        __shared__ float s_act[TILE_M_SM][QK];

        if (tm < TILE_M_SM && tn < QK) {
            if (m_base + tm < M) {
                s_act[tm][tn] = activation[m_base * K + kb * QK + tn];
            }
        }
        __syncthreads();

        if (tm < TILE_M_SM && n_base < N) {
            acc += dot_q4_0_fp32_block(
                &weight[(n_base * num_blocks + kb) * Q4_0_BYTES],
                s_act[tm]
            );
        }
        __syncthreads();
    }

    if (tm < TILE_M_SM && m_base < M && n_base < N) {
        output[m_base * N + n_base] = acc;
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 8) {
        int n_per_warp = N_PER_WARP;
        dim3 block(32);
        dim3 grid((N + n_per_warp - 1) / n_per_warp, M);
        gemm_small_dp4a<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 block(TILE_N_SM, TILE_M_SM);
        dim3 grid((N + TILE_N_SM - 1) / TILE_N_SM, (M + TILE_M_SM - 1) / TILE_M_SM);
        gemm_large_shared<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V7 - Best Pattern Adapted)");
}
