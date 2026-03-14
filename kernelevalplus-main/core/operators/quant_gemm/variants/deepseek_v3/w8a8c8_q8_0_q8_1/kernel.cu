#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK8_0 32
#define QK8_1 32
#define Q8_0_BLOCK_SIZE 34
#define Q8_1_BLOCK_SIZE 36

// DP4A instruction wrapper (matching llama.cpp style)
__device__ __forceinline__ int dp4a(const int a, const int b, const int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// vec_dot_q8_0_q8_1_impl from llama.cpp
template <int vdr>
__device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {

    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values (matching llama.cpp)
        sumi = dp4a(v[i], u[i], sumi);
    }

    return d8_0 * d8_1 * ((float) sumi);
}

__global__ void gemm_w8a8c8_q8_0_q8_1_kernel(
    const uint8_t* weight,     // [N, K/32, 34] - Q8_0 quantized weights
    const float* activation,   // [M, K] - FP32 activations
    float* output,             // [M, N] - FP32 output
    int M,
    int N,
    int K
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int num_blocks = K / QK8_0;

    // Process activation in blocks (Q8_1 style per-block quantization)
    for (int kb = 0; kb < num_blocks; ++kb) {
        // === Load Q8_0 weight block ===
        long long w_offset = ((long long)n * num_blocks + kb) * Q8_0_BLOCK_SIZE;

        // Read Q8_0 scale (safe unaligned access using union)
        union {
            uint16_t u16;
            __half f16;
        } scale_union;
        scale_union.u16 = ((uint16_t)weight[w_offset + 0]) | (((uint16_t)weight[w_offset + 1]) << 8);
        float d_w = __half2float(scale_union.f16);

        // === Quantize activation block to Q8_1 style ===
        int k_start = kb * QK8_1;

        // Find activation scale for this block (Q8_1 style)
        float a_max = 0.0f;
        for (int i = 0; i < QK8_1; ++i) {
            float a_val = activation[m * K + k_start + i];
            a_max = fmaxf(a_max, fabsf(a_val));
        }
        float d_a = a_max / 127.0f;
        if (d_a < 1e-6f) d_a = 1.0f;

        // Pack quantized activation values for DP4A
        int a_qs_int[QK8_1 / 4];  // 8 ints for 32 int8 values
        #pragma unroll
        for (int i = 0; i < QK8_1 / 4; ++i) {
            int8_t tmp[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int k_global = k_start + i * 4 + j;
                float a_val = activation[m * K + k_global];
                int a_int32 = (int)roundf(a_val / d_a);
                a_int32 = (a_int32 < -128) ? -128 : ((a_int32 > 127) ? 127 : a_int32);
                tmp[j] = (int8_t)a_int32;
            }
            // Pack 4 int8 values into 1 int (little-endian)
            a_qs_int[i] = (int)tmp[0] | ((int)tmp[1] << 8) |
                         ((int)tmp[2] << 16) | ((int)tmp[3] << 24);
        }

        // Pack weight values for DP4A (safe unaligned access)
        int w_qs_int[QK8_0 / 4];  // 8 ints for 32 int8 values
        #pragma unroll
        for (int i = 0; i < QK8_0 / 4; ++i) {
            // Use byte-by-byte packing to avoid alignment issues
            int8_t tmp[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                tmp[j] = (int8_t)weight[w_offset + 2 + i * 4 + j];
            }
            // Pack 4 int8 values into 1 int (little-endian)
            w_qs_int[i] = (int)tmp[0] | ((int)tmp[1] << 8) |
                         ((int)tmp[2] << 16) | ((int)tmp[3] << 24);
        }

        // === INT8 dot product using llama.cpp style ===
        // VDR_Q8_0_Q8_1_MMQ = 8 (32 int8 values = 8 int32 values)
        constexpr int vdr = 8;
        sum += vec_dot_q8_0_q8_1_impl<vdr>(w_qs_int, a_qs_int, d_w, d_a);
    }

    output[m * N + n] = sum;
}

extern "C" void gemm_w8a8c8_q8_0_q8_1(
    const uint8_t* weight,
    const float* activation,
    float* output,
    int M,
    int N,
    int K
) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    gemm_w8a8c8_q8_0_q8_1_kernel<<<gridDim, blockDim>>>(weight, activation, output, M, N, K);
}
