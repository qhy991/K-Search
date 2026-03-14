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
    // Q4_0 packs 4-bit values: each byte contains 2 quants
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int8_t tl[4], th[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t b = w_ptr[2 + i*4 + j];
            tl[j] = (int8_t)(b & 0x0F);          // low nibble
            th[j] = (int8_t)((b >> 4) & 0x0F);   // high nibble
        }
        sumi = dp4a_i8(*reinterpret_cast<int*>(tl), *reinterpret_cast<int*>(&aq[i*4]), sumi);
        sumi = dp4a_i8(*reinterpret_cast<int*>(th), *reinterpret_cast<int*>(&aq[16+i*4]), sumi);
    }

    // Apply scales: result = d_w * (d_a * sumi - 8 * a_sum)
    return d_w * (d_a * (float)sumi - 8.0f * a_sum);
}

// Small M kernel - optimized for memory-bound regime
// Each warp computes one N output across threads
#define K_THREADS_SMALL 8

__global__ void gemm_small_m(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;  // K=2048 -> 64 blocks

    // Warp-level division of work
    int lane = threadIdx.x;  // 0-31
    int n_local = lane / K_THREADS_SMALL;      // N index within warp
    int k_part = lane % K_THREADS_SMALL;       // K partition within warp
    int n_per_warp = 32 / K_THREADS_SMALL;     // N outputs per warp

    int n = blockIdx.x * n_per_warp + n_local;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    // Partition K blocks among threads
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

    // Warp reduction to sum partial results
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int off = 1; off < K_THREADS_SMALL; off *= 2) {
        partial += __shfl_down_sync(mask, partial, off);
    }

    // First thread in K group writes result
    if (k_part == 0 && n < N) {
        output[m * N + n] = partial;
    }
}

// Large M kernel with shared memory tiling
// Optimized for compute-bound regime with larger batches
#define TILE_M_SM 8
#define TILE_N_SM 64

__global__ void gemm_large_m_shared(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;  // 64 blocks for K=2048

    // Thread indices
    int tm = threadIdx.y;  // 0 to TILE_M_SM-1 (row in tile)
    int tn = threadIdx.x;  // 0 to TILE_N_SM-1 (col in tile)

    // Global indices
    int m_base = blockIdx.y * TILE_M_SM + tm;
    int n_base = blockIdx.x * TILE_N_SM + tn;

    if (m_base >= M || n_base >= N) return;

    float acc = 0.0f;

    // Process each K block
    for (int kb = 0; kb < num_blocks; kb++) {
        // Load activation block into shared memory
        // Each row in the tile loads its 32 activation values
        __shared__ float s_act[TILE_M_SM][QK];

        if (tm < TILE_M_SM && tn < QK) {
            if (m_base < M) {
                s_act[tm][tn] = activation[m_base * K + kb * QK + tn];
            } else {
                s_act[tm][tn] = 0.0f;
            }
        }
        __syncthreads();

        // Compute dot product for this K block
        if (tm < TILE_M_SM && m_base < M) {
            acc += dot_q4_0_q8_1_block(
                &weight[((int64_t)n_base * num_blocks + kb) * Q4_0_BYTES],
                s_act[tm]
            );
        }
        __syncthreads();
    }

    if (tm < TILE_M_SM && m_base < M) {
        output[m_base * N + n_base] = acc;
    }
}

// Medium M kernel - balanced approach for intermediate batch sizes
// Uses vectorized loads but no shared memory
#define VEC_N_MID 32  // Process 32 N outputs per warp

__global__ void gemm_medium_m(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;  // 64 blocks

    // Each warp processes VEC_N_MID outputs
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int n_base = warp_id * VEC_N_MID + lane / 4;
    int m_base = blockIdx.y;

    if (m_base >= M || n_base >= N) return;

    // Partition K work among groups of 4 threads
    int k_part = lane % 4;
    int bk_per = (num_blocks + 4 - 1) / 4;
    int b_start = k_part * bk_per;
    int b_end = min(b_start + bk_per, num_blocks);

    float partial = 0.0f;
    for (int b = b_start; b < b_end; ++b) {
        partial += dot_q4_0_q8_1_block(
            &weight[((int64_t)n_base * num_blocks + b) * Q4_0_BYTES],
            &activation[m_base * K + b * QK]
        );
    }

    // Reduce across 4 threads
    #pragma unroll
    for (int off = 1; off < 4; off *= 2) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
    }

    if ((lane & 3) == 0 && n_base < N) {
        output[m_base * N + n_base] = partial;
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

    // Strategy dispatch based on M (batch size)
    if (M <= 8) {
        // Small batch: memory-bound, use warp-centric kernel
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
        // Medium batch: balanced approach
        dim3 block(32, 2);  // 32 threads x 2 warps per block
        int warps_needed = (N + VEC_N_MID - 1) / VEC_N_MID * M;
        dim3 grid((warps_needed + 2 - 1) / 2);
        gemm_medium_m<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
        err = cudaGetLastError();
    } else {
        // Large batch: compute-bound, use shared memory tiling
        dim3 block(TILE_N_SM, TILE_M_SM);
        dim3 grid((N + TILE_N_SM - 1) / TILE_N_SM, (M + TILE_M_SM - 1) / TILE_M_SM);
        gemm_large_m_shared<<<grid, block>>>(
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
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation) - ds3 moe routing up");
}
