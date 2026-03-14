#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK 32
#define Q4_0_BYTES 18
#define K_THREADS 8

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
    float d_w = read_fp16(w_ptr);

    float amax = 0.0f;
    float a_vals[QK];
    float a_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        float v = a_ptr[i];
        a_vals[i] = v;
        a_sum += v;
        amax = fmaxf(amax, fabsf(v));
    }

    float d_a = amax / 127.0f;
    if (d_a < 1e-10f) d_a = 1.0f;
    float inv_d = 1.0f / d_a;

    int8_t aq[QK];
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        int v = __float2int_rn(a_vals[i] * inv_d);
        aq[i] = (int8_t)max(-128, min(127, v));
    }

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

// Small M kernel with K-parallelization
__global__ void gemm_small_m(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;
    int lane = threadIdx.x;
    int n_local = lane / K_THREADS;
    int k_part = lane % K_THREADS;
    int n_per_warp = 32 / K_THREADS;
    int n = blockIdx.x * n_per_warp + n_local;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    int bk_per = (num_blocks + K_THREADS - 1) / K_THREADS;
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
    for (int off = 1; off < K_THREADS; off *= 2) {
        partial += __shfl_down_sync(mask, partial, off);
    }

    if (k_part == 0 && n < N) {
        output[m * N + n] = partial;
    }
}

// Large M kernel - simple 1-thread-per-element
__global__ void gemm_large_m(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ activation,
    float*         __restrict__ output,
    int M, int N, int K
) {
    int num_blocks = K / QK;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    float sum = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        sum += dot_q4_0_q8_1_block(
            &weight[((int64_t)n * num_blocks + b) * Q4_0_BYTES],
            &activation[m * K + b * QK]
        );
    }
    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 16) {
        // Small batch: K-parallelization kernel
        int n_per_warp = 32 / K_THREADS;
        dim3 block(32);
        dim3 grid((N + n_per_warp - 1) / n_per_warp, M);
        gemm_small_m<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)M, (int)N, (int)K
        );
    } else {
        // Large batch: tuned block size for M=512
        // Use 512 threads for good balance between occupancy and block count
        dim3 block(512);
        dim3 grid((N + 511) / 512, M);
        gemm_large_m<<<grid, block>>>(
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
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation)");
}
