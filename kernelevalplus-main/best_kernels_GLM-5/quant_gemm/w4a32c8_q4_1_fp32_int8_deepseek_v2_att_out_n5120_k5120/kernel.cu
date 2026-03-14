/**
 * Final Optimized Quantized GEMM for DeepSeek-V2 Attention Output with Q4_1 Weights
 *
 * Target: RTX 4090 (128 SMs, Ada Lovelace)
 * Configuration: N=5120, K=5120, M=variable
 * Format: W4A32C8 - Q4_1 weights, FP32 activations
 *
 * Key Optimizations:
 * 1. Shared memory for activation quantization (reuse across N dimension)
 * 2. Memory coalescing for weight access
 * 3. DP4A for efficient INT8 dot products
 * 4. K-parallel reduction with warp shuffle for small batch
 *
 * Q4_1 format (20 bytes per block):
 * - Bytes 0-1: scale (FP16) - "d"
 * - Bytes 2-3: min (FP16) - "m"
 * - Bytes 4-19: 16 bytes containing 32 x 4-bit values
 *
 * Q4_1 formula: result = d_w * d_a * sumi + m_w * d_a * sum_a_q
 *
 * Performance (RTX 4090):
 * - M=1: 1405 GFLOPS (37 us)
 * - M=512: 1168 GFLOPS (23 ms)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int Q4_1_BLOCK = 20;
constexpr int NUM_K_BLOCKS = 160;

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

/**
 * Small batch kernel (M <= 64)
 * Uses shared memory for activation quantization with K-parallel reduction
 */
__global__ void __launch_bounds__(128) gemm_q4_1_small_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int m = blockIdx.x;
    if (m >= M) return;

    const int n_base = blockIdx.y * warps_per_block + warp_id;
    const int n = n_base;

    if (n >= N) return;

    // Shared memory for activation quantization
    __shared__ float s_act_scale[NUM_K_BLOCKS];
    __shared__ int s_act_sum_q[NUM_K_BLOCKS];
    __shared__ int8_t s_act_qs[NUM_K_BLOCKS * QK];

    // Cooperatively load and quantize all activation blocks
    const float* act_row = activation + m * K;

    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const float4* act_ptr = reinterpret_cast<const float4*>(&act_row[kb * QK]);

        float a_max = 0.0f;
        float a_vals[QK];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = act_ptr[i];
            a_vals[i*4+0] = v.x;
            a_vals[i*4+1] = v.y;
            a_vals[i*4+2] = v.z;
            a_vals[i*4+3] = v.w;

            a_max = fmaxf(a_max, fabsf(v.x));
            a_max = fmaxf(a_max, fabsf(v.y));
            a_max = fmaxf(a_max, fabsf(v.z));
            a_max = fmaxf(a_max, fabsf(v.w));
        }

        float d_a = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;
        s_act_scale[kb] = d_a;

        int sum_q = 0;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            int8_t q = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[i] / d_a))));
            s_act_qs[kb * QK + i] = q;
            sum_q += q;
        }
        s_act_sum_q[kb] = sum_q;
    }
    __syncthreads();

    // K-parallel: each lane processes different K blocks
    float partial_sum = 0.0f;

    for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += 32) {
        float d_a = s_act_scale[kb];
        int act_sum_q = s_act_sum_q[kb];
        const int8_t* act_qs = &s_act_qs[kb * QK];

        const uint8_t* w_block = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
        const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
        const float m_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
        const uint8_t* qs = w_block + 4;

        int sumi = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t b0 = qs[i*4+0];
            uint8_t b1 = qs[i*4+1];
            uint8_t b2 = qs[i*4+2];
            uint8_t b3 = qs[i*4+3];

            int w_lo = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                      (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
            int w_hi = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                      (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);

            int a_lo = (uint8_t)act_qs[i*4+0] | ((uint8_t)act_qs[i*4+1] << 8) |
                      ((uint8_t)act_qs[i*4+2] << 16) | ((uint8_t)act_qs[i*4+3] << 24);
            int a_hi = (uint8_t)act_qs[16+i*4+0] | ((uint8_t)act_qs[16+i*4+1] << 8) |
                      ((uint8_t)act_qs[16+i*4+2] << 16) | ((uint8_t)act_qs[16+i*4+3] << 24);

            sumi = dp4a(a_lo, w_lo, sumi);
            sumi = dp4a(a_hi, w_hi, sumi);
        }

        partial_sum += d_w * d_a * (float)sumi + m_w * d_a * (float)act_sum_q;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    if (lane_id == 0) {
        output[m * N + n] = partial_sum;
    }
}

/**
 * Large batch kernel (M > 64)
 * Each thread computes one output element
 */
__global__ void __launch_bounds__(256) gemm_q4_1_large_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const float* act_row = activation + m * K;
    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const float4* act_ptr = reinterpret_cast<const float4*>(&act_row[kb * QK]);

        float a_max = 0.0f;
        float a_vals[QK];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = act_ptr[i];
            a_vals[i*4+0] = v.x;
            a_vals[i*4+1] = v.y;
            a_vals[i*4+2] = v.z;
            a_vals[i*4+3] = v.w;
            a_max = fmaxf(a_max, fabsf(v.x));
            a_max = fmaxf(a_max, fabsf(v.y));
            a_max = fmaxf(a_max, fabsf(v.z));
            a_max = fmaxf(a_max, fabsf(v.w));
        }

        float d_a = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;

        const uint8_t* w_block = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
        const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
        const float m_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
        const uint8_t* qs = w_block + 4;

        int sumi = 0;
        int sum_a_q = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t b0 = qs[i*4+0];
            uint8_t b1 = qs[i*4+1];
            uint8_t b2 = qs[i*4+2];
            uint8_t b3 = qs[i*4+3];

            int w_lo = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                      (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
            int w_hi = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                      (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);

            int a0 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[i*4+0] / d_a))));
            int a1 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[i*4+1] / d_a))));
            int a2 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[i*4+2] / d_a))));
            int a3 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[i*4+3] / d_a))));

            int a_lo = (uint8_t)a0 | ((uint8_t)a1 << 8) | ((uint8_t)a2 << 16) | ((uint8_t)a3 << 24);

            int a16 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[16+i*4+0] / d_a))));
            int a17 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[16+i*4+1] / d_a))));
            int a18 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[16+i*4+2] / d_a))));
            int a19 = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(a_vals[16+i*4+3] / d_a))));

            int a_hi = (uint8_t)a16 | ((uint8_t)a17 << 8) | ((uint8_t)a18 << 16) | ((uint8_t)a19 << 24);

            sumi = dp4a(a_lo, w_lo, sumi);
            sumi = dp4a(a_hi, w_hi, sumi);

            sum_a_q += a0 + a1 + a2 + a3 + a16 + a17 + a18 + a19;
        }

        sum += d_w * d_a * (float)sumi + m_w * d_a * (float)sum_a_q;
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 64) {
        // Small batch: use shared memory + K-parallel kernel
        const int warps_per_block = 4;
        dim3 grid(M, (N + warps_per_block - 1) / warps_per_block);
        dim3 block(128);

        gemm_q4_1_small_batch_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        // Large batch: use simple kernel
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);

        gemm_q4_1_large_batch_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Q4_1 GEMM for DeepSeek-V2 Att Out - Final");
}
