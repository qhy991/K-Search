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

// Basic dot product (for large M - lower register pressure)
__device__ __forceinline__ float dot_q4_0_q8_1_block(
    const uint8_t* __restrict__ w_ptr,
    const float* __restrict__ a_ptr
) {
    float d_w = read_fp16(w_ptr);

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

    int8_t aq[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int v = __float2int_rn(a_ptr[i] * inv_d);
        aq[i] = (int8_t)max(-128, min(127, v));
    }

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

    return d_w * (d_a * (float)sumi - 8.0f * a_sum);
}

// ILP-optimized dot product (for small M - higher ILP)
// Processes 2 K blocks per call
__device__ __forceinline__ void dot_q4_0_q8_1_block_accum(
    const uint8_t* __restrict__ w_ptr0,
    const uint8_t* __restrict__ w_ptr1,
    const float* __restrict__ a_ptr0,
    const float* __restrict__ a_ptr1,
    float& acc0,
    float& acc1
) {
    float d_w0 = read_fp16(w_ptr0);
    float d_w1 = read_fp16(w_ptr1);

    float amax0 = 0.0f, amax1 = 0.0f;
    float a_sum0 = 0.0f, a_sum1 = 0.0f;

    #pragma unroll
    for (int i = 0; i < QK; i++) {
        float v0 = a_ptr0[i], v1 = a_ptr1[i];
        a_sum0 += v0; a_sum1 += v1;
        amax0 = fmaxf(amax0, fabsf(v0));
        amax1 = fmaxf(amax1, fabsf(v1));
    }

    float d_a0 = amax0 / 127.0f;
    float d_a1 = amax1 / 127.0f;
    if (d_a0 < 1e-10f) d_a0 = 1.0f;
    if (d_a1 < 1e-10f) d_a1 = 1.0f;
    float inv_d0 = 1.0f / d_a0;
    float inv_d1 = 1.0f / d_a1;

    int8_t aq0[32], aq1[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int v0 = __float2int_rn(a_ptr0[i] * inv_d0);
        int v1 = __float2int_rn(a_ptr1[i] * inv_d1);
        aq0[i] = (int8_t)max(-128, min(127, v0));
        aq1[i] = (int8_t)max(-128, min(127, v1));
    }

    int sumi0 = 0, sumi1 = 0;
    const uint8_t* w_qs0 = w_ptr0 + 2;
    const uint8_t* w_qs1 = w_ptr1 + 2;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int8_t tl0[4], th0[4], tl1[4], th1[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t b0 = w_qs0[i*4 + j];
            uint8_t b1 = w_qs1[i*4 + j];
            tl0[j] = (int8_t)(b0 & 0x0F);
            th0[j] = (int8_t)((b0 >> 4) & 0x0F);
            tl1[j] = (int8_t)(b1 & 0x0F);
            th1[j] = (int8_t)((b1 >> 4) & 0x0F);
        }
        sumi0 = dp4a_i8(*reinterpret_cast<int*>(tl0), *reinterpret_cast<int*>(&aq0[i*4]), sumi0);
        sumi0 = dp4a_i8(*reinterpret_cast<int*>(th0), *reinterpret_cast<int*>(&aq0[16+i*4]), sumi0);
        sumi1 = dp4a_i8(*reinterpret_cast<int*>(tl1), *reinterpret_cast<int*>(&aq1[i*4]), sumi1);
        sumi1 = dp4a_i8(*reinterpret_cast<int*>(th1), *reinterpret_cast<int*>(&aq1[16+i*4]), sumi1);
    }

    acc0 += d_w0 * (d_a0 * (float)sumi0 - 8.0f * a_sum0);
    acc1 += d_w1 * (d_a1 * (float)sumi1 - 8.0f * a_sum1);
}

// Small M kernel - ILP optimized (use paired processing)
#define K_THREADS_SMALL 8

__global__ void gemm_small_m_ilp(
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
    int b = b_start;
    for (; b + 1 < b_end; b += 2) {
        float acc0 = 0.0f, acc1 = 0.0f;
        dot_q4_0_q8_1_block_accum(
            &weight[((int64_t)n * num_blocks + b) * Q4_0_BYTES],
            &weight[((int64_t)n * num_blocks + b + 1) * Q4_0_BYTES],
            &activation[m * K + b * QK],
            &activation[m * K + (b + 1) * QK],
            acc0, acc1
        );
        partial += acc0 + acc1;
    }
    // Handle odd number of blocks
    if (b < b_end) {
        float d_w = read_fp16(&weight[((int64_t)n * num_blocks + b) * Q4_0_BYTES]);

        float amax = 0.0f;
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float v = activation[m * K + b * QK + i];
            a_sum += v;
            amax = fmaxf(amax, fabsf(v));
        }

        float d_a = amax / 127.0f;
        if (d_a < 1e-10f) d_a = 1.0f;
        float inv_d = 1.0f / d_a;

        int8_t aq[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int v = __float2int_rn(activation[m * K + b * QK + i] * inv_d);
            aq[i] = (int8_t)max(-128, min(127, v));
        }

        int sumi = 0;
        const uint8_t* w_qs = &weight[((int64_t)n * num_blocks + b) * Q4_0_BYTES] + 2;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int8_t tl[4], th[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t byte = w_qs[i*4 + j];
                tl[j] = (int8_t)(byte & 0x0F);
                th[j] = (int8_t)((byte >> 4) & 0x0F);
            }
            sumi = dp4a_i8(*reinterpret_cast<int*>(tl), *reinterpret_cast<int*>(&aq[i*4]), sumi);
            sumi = dp4a_i8(*reinterpret_cast<int*>(th), *reinterpret_cast<int*>(&aq[16+i*4]), sumi);
        }

        partial += d_w * (d_a * (float)sumi - 8.0f * a_sum);
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

// Large M kernel - simple direct access (lower register pressure)
#define TILE_N_LG 512
#define TILE_M_LG 1

__global__ void gemm_large_m_simple(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;
    int m = blockIdx.y * TILE_M_LG + threadIdx.y;
    int n = blockIdx.x * TILE_N_LG + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int kb = 0; kb < num_blocks; kb++) {
        acc += dot_q4_0_q8_1_block(
            &weight[((int64_t)n * num_blocks + kb) * Q4_0_BYTES],
            &activation[m * K + kb * QK]
        );
    }

    output[m * N + n] = acc;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    // Strategy dispatch based on performance results
    // Small M (<=8): Use ILP-optimized kernel - best for single token
    // Large M (>8): Use simple kernel - better for large batches (less register pressure)

    if (M <= 8) {
        // Very small M: ILP-optimized for best single-token performance
        int n_per_warp = 32 / K_THREADS_SMALL;
        dim3 block(32);
        dim3 grid((N + n_per_warp - 1) / n_per_warp, M);
        gemm_small_m_ilp<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
    } else if (M <= 16) {
        // Small M: Use simple kernel (v1 approach) - balanced
        int n_per_warp = 32 / K_THREADS_SMALL;
        dim3 block(32);
        dim3 grid((N + n_per_warp - 1) / n_per_warp, M);
        gemm_small_m_ilp<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
    } else {
        // Large M: Use simple direct access kernel
        // Lower register pressure helps with occupancy for large batches
        dim3 block(TILE_N_LG, TILE_M_LG);
        dim3 grid((N + TILE_N_LG - 1) / TILE_N_LG, (M + TILE_M_LG - 1) / TILE_M_LG);
        gemm_large_m_simple<<<grid, block>>>(
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
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation) - Combined Best");
}
