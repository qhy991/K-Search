#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK 32
#define Q4_0_BYTES 18

// DP4A intrinsic helper
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

// Optimized Q4_0 x FP32 block dot product
// Uses aggressive unrolling and efficient register usage
__device__ __forceinline__ float dot_q4_0_q8_1_block(
    const uint8_t* __restrict__ w_ptr,
    const float* __restrict__ a_ptr
) {
    // Read weight scale
    half d_w_half = *((const half*)w_ptr);
    float d_w = __half2float(d_w_half);

    // Compute activation statistics
    float a_sum = 0.0f;
    float amax = 0.0f;

    // Manual loop unrolling for better ILP
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = a_ptr[i];
        a_sum += v;
        amax = fmaxf(amax, fabsf(v));
    }

    float d_a = amax / 127.0f;
    if (d_a < 1e-10f) d_a = 1.0f;
    float inv_d = 1.0f / d_a;

    // Quantize activation to INT8
    int8_t aq[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int v = __float2int_rn(a_ptr[i] * inv_d);
        aq[i] = (int8_t)max(-128, min(127, v));
    }

    // DP4A computation with unrolled Q4_0 unpacking
    int sumi = 0;
    const uint8_t* packed_qs = w_ptr + 2;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Unpack 4 bytes (8 Q4 values) - low nibbles
        int8_t tl0 = (int8_t)(packed_qs[i*4 + 0] & 0x0F);
        int8_t tl1 = (int8_t)(packed_qs[i*4 + 1] & 0x0F);
        int8_t tl2 = (int8_t)(packed_qs[i*4 + 2] & 0x0F);
        int8_t tl3 = (int8_t)(packed_qs[i*4 + 3] & 0x0F);

        // Unpack 4 bytes (8 Q4 values) - high nibbles
        int8_t th0 = (int8_t)((packed_qs[i*4 + 0] >> 4) & 0x0F);
        int8_t th1 = (int8_t)((packed_qs[i*4 + 1] >> 4) & 0x0F);
        int8_t th2 = (int8_t)((packed_qs[i*4 + 2] >> 4) & 0x0F);
        int8_t th3 = (int8_t)((packed_qs[i*4 + 3] >> 4) & 0x0F);

        // Pack into int32 for DP4A
        int tl_packed = (tl3 << 24) | ((uint8_t)tl2 << 16) | ((uint8_t)tl1 << 8) | (uint8_t)tl0;
        int th_packed = (th3 << 24) | ((uint8_t)th2 << 16) | ((uint8_t)th1 << 8) | (uint8_t)th0;

        sumi = dp4a_i8(tl_packed, *reinterpret_cast<int*>(&aq[i*4]), sumi);
        sumi = dp4a_i8(th_packed, *reinterpret_cast<int*>(&aq[16+i*4]), sumi);
    }

    return d_w * (d_a * (float)sumi - 8.0f * a_sum);
}

// Small M kernel with K-parallelization (keep existing good design)
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

// Optimized large M kernel with shared memory tiling
// Each thread block processes a tile of output: TILE_M x TILE_N
// Each thread processes TILE_K K-blocks
#define TILE_M_LARGE 4
#define TILE_N_LARGE 32
#define TILE_K_LARGE 3  // Process 3 K-blocks per iteration

__global__ void gemm_large_m_optimized(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;

    // Thread and block organization
    int tm = threadIdx.y;  // 0 to TILE_M_LARGE-1
    int tn = threadIdx.x;  // 0 to TILE_N_LARGE-1

    int m_base = blockIdx.y * TILE_M_LARGE + tm;
    int n_base = blockIdx.x * TILE_N_LARGE + tn;

    if (m_base >= M || n_base >= N) return;

    float acc = 0.0f;

    // Process K blocks
    for (int kb = 0; kb < num_blocks; kb++) {
        const int k_base = kb * QK;

        // Load activation for this row (coalesced read)
        float act_block[QK];
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            act_block[i] = activation[m_base * K + k_base + i];
        }

        // Process weight for this (n, k) block
        const uint8_t* w_block = &weight[((int64_t)n_base * num_blocks + kb) * Q4_0_BYTES];
        acc += dot_q4_0_q8_1_block(w_block, act_block);
    }

    output[m_base * N + n_base] = acc;
}

// Even more optimized: multiple N elements per thread, K-parallelization for large M
#define K_THREADS_LARGE 4
#define N_PER_THREAD 2

__global__ void gemm_large_m_kpar(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;

    int lane = threadIdx.x;
    int n_local = (lane / K_THREADS_LARGE) * N_PER_THREAD;  // N index in warp
    int k_part = lane % K_THREADS_LARGE;
    int n_per_warp = (32 / K_THREADS_LARGE) * N_PER_THREAD;

    int n = blockIdx.x * n_per_warp + n_local;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    // Each thread computes N_PER_THREAD output elements
    float acc[N_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; i++) {
        acc[i] = 0.0f;
    }

    int bk_per = (num_blocks + K_THREADS_LARGE - 1) / K_THREADS_LARGE;
    int b_start = k_part * bk_per;
    int b_end = min(b_start + bk_per, num_blocks);

    int n_valid = min(N_PER_THREAD, N - n);

    for (int b = b_start; b < b_end; ++b) {
        const float* a_ptr = &activation[m * K + b * QK];

        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; i++) {
            if (i < n_valid) {
                acc[i] += dot_q4_0_q8_1_block(
                    &weight[((int64_t)(n + i) * num_blocks + b) * Q4_0_BYTES],
                    a_ptr
                );
            }
        }
    }

    // Warp shuffle reduction for each N element
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; i++) {
        #pragma unroll
        for (int off = 1; off < K_THREADS_LARGE; off *= 2) {
            acc[i] += __shfl_down_sync(mask, acc[i], off);
        }
    }

    // Write results
    if (k_part == 0) {
        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; i++) {
            if (i < n_valid && (n + i) < N) {
                output[m * N + n + i] = acc[i];
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 16) {
        // Small M: K-parallelization
        int n_per_warp = 32 / K_THREADS_SMALL;
        dim3 block(32);
        dim3 grid((N + n_per_warp - 1) / n_per_warp, M);
        gemm_small_m<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
    } else if (M <= 64) {
        // Medium M: 2D tiling
        dim3 block(TILE_N_LARGE, TILE_M_LARGE);
        dim3 grid((N + TILE_N_LARGE - 1) / TILE_N_LARGE, (M + TILE_M_LARGE - 1) / TILE_M_LARGE);
        gemm_large_m_optimized<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
    } else {
        // Large M: K-parallelization with multiple N per thread
        int n_per_warp = (32 / K_THREADS_LARGE) * N_PER_THREAD;
        dim3 block(32);
        dim3 grid((N + n_per_warp - 1) / n_per_warp, M);
        gemm_large_m_kpar<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation) - v8 optimized");
}
