#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

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

__device__ __forceinline__ float dot_q4_0_q8_1_block(
    const uint8_t* __restrict__ w_ptr,
    const float* __restrict__ a_ptr
) {
    // Q4_0 format: scale (fp16, 2 bytes) + 16 quants (16 bytes) = 18 bytes
    float d_w = read_fp16(w_ptr);

    // Dynamic Q8_1 quantization for activation
    float amax = 0.0f;
    float a_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        float v = a_ptr[i];
        a_sum += v;
        amax = fmaxf(amax, fabsf(v));
    }

    float d_a = amax / 127.0f;
    if (d_a < 1e-10f) d_a = 1.0f;
    float inv_d = 1.0f / d_a;

    // Quantize activation to int8
    int8_t aq[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int v = __float2int_rn(a_ptr[i] * inv_d);
        aq[i] = (int8_t)max(-128, min(127, v));
    }

    // Compute dot product with Q4_0 weights
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int8_t tl[4], th[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t b = w_ptr[2 + i*4 + j];
            tl[j] = (int8_t)(b & 0x0F);
            th[j] = (int8_t)((b >> 4) & 0x0F);
        }
        sumi = dp4a_i8(*reinterpret_cast<int*>(tl), *reinterpret_cast<int*>(&aq[i*4]), sumi);
        sumi = dp4a_i8(*reinterpret_cast<int*>(th), *reinterpret_cast<int*>(&aq[16+i*4]), sumi);
    }

    return d_w * (d_a * (float)sumi - 8.0f * a_sum);
}

// Small M kernel - optimized for memory-bound regime
#define K_THREADS_SMALL 8

__global__ void gemm_small_m(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;

    int lane = threadIdx.x;
    int n_local = lane / K_THREADS_SMALL;
    int k_part = lane % K_THREADS_SMALL;
    int n_per_warp = 32 / K_THREADS_SMALL;

    int n = blockIdx.x * n_per_warp + n_local;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    int bk_per = (num_blocks + K_THREADS_SMALL - 1) / K_THREADS_SMALL;
    int b_start = k_part * bk_per;
    int b_end = min(b_start + bk_per, num_blocks);

    float partial = 0.0f;
    for (int b = b_start; b < b_end; ++b) {
        partial += dot_q4_0_q8_1_block(
            &weight[((int64_t)n * num_blocks + b) * Q4_0_BYTES],
            &activation[m * K + b * QK]
        );
    }

    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int off = 1; off < K_THREADS_SMALL; off *= 2) {
        partial += __shfl_down_sync(mask, partial, off);
    }

    if (k_part == 0 && n < N) {
        output[m * N + n] = partial;
    }
}

// Large M kernel - optimized for compute-bound with better ILP
// Each warp processes multiple N outputs to hide latency
#define K_PER_THREAD_LARGE 4

__global__ void gemm_large_m_ilp(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;

    // Each thread computes K_PER_THREAD_LARGE different N outputs
    int lane = threadIdx.x;
    int n_base = (blockIdx.x * 32 + lane) * K_PER_THREAD_LARGE;
    int m = blockIdx.y;

    if (m >= M || n_base >= N) return;

    // Process all K blocks
    float partials[K_PER_THREAD_LARGE];
    #pragma unroll
    for (int p = 0; p < K_PER_THREAD_LARGE; p++) {
        partials[p] = 0.0f;
    }

    for (int b = 0; b < num_blocks; b++) {
        #pragma unroll
        for (int p = 0; p < K_PER_THREAD_LARGE; p++) {
            int n = n_base + p;
            if (n < N) {
                partials[p] += dot_q4_0_q8_1_block(
                    &weight[((int64_t)n * num_blocks + b) * Q4_0_BYTES],
                    &activation[m * K + b * QK]
                );
            }
        }
    }

    #pragma unroll
    for (int p = 0; p < K_PER_THREAD_LARGE; p++) {
        int n = n_base + p;
        if (n < N) {
            output[m * N + n] = partials[p];
        }
    }
}

// Ultra-large M kernel with 2D tiling for maximum compute throughput
// Processes multiple output values simultaneously to maximize ILP
#define TILE_N_SM 64
#define TILE_K_SM 2

__global__ void gemm_ultra_large_m_2d(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;

    // Each thread processes TILE_K_SM x 1 outputs
    int lane = threadIdx.x;
    int n_base = blockIdx.x * TILE_N_SM + lane % TILE_N_SM;
    int k_base = (lane / TILE_N_SM) * TILE_K_SM;

    int m = blockIdx.y;

    if (m >= M || n_base >= N) return;

    float partials[TILE_K_SM];
    #pragma unroll
    for (int p = 0; p < TILE_K_SM; p++) {
        partials[p] = 0.0f;
    }

    for (int b = k_base; b < num_blocks; b += TILE_K_SM) {
        #pragma unroll
        for (int p = 0; p < TILE_K_SM; p++) {
            int kb = b + p;
            if (kb < num_blocks) {
                partials[p] += dot_q4_0_q8_1_block(
                    &weight[((int64_t)n_base * num_blocks + kb) * Q4_0_BYTES],
                    &activation[m * K + kb * QK]
                );
            }
        }
    }

    // Sum across K dimension partitions
    float result = 0.0f;
    #pragma unroll
    for (int p = 0; p < TILE_K_SM; p++) {
        result += partials[p];
    }

    if (n_base < N) {
        output[m * N + n_base] = result;
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    cudaError_t err;

    // Strategy dispatch based on M
    if (M <= 8) {
        // Small batch: memory-bound
        int n_per_warp = 32 / K_THREADS_SMALL;
        dim3 block(32);
        dim3 grid((N + n_per_warp - 1) / n_per_warp, M);
        gemm_small_m<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
        err = cudaGetLastError();
    } else if (M <= 64) {
        // Medium batch: single-thread multi-output ILP
        dim3 block(32);
        dim3 grid((N + 31 * K_PER_THREAD_LARGE) / (32 * K_PER_THREAD_LARGE), M);
        gemm_large_m_ilp<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
        err = cudaGetLastError();
    } else {
        // Large batch: 2D tiling for maximum compute throughput
        dim3 block(TILE_N_SM * 2, 1);  // 2 groups of TILE_N_SM threads
        dim3 grid((N + TILE_N_SM - 1) / TILE_N_SM, M);
        gemm_ultra_large_m_2d<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
        err = cudaGetLastError();
    }

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation) - ds3 moe routing up v2 with ILP");
}
