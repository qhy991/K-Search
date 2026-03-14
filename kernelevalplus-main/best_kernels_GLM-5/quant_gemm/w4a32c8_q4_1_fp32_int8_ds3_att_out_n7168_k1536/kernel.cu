/**
 * W4A32C8 Q4_1 Quantized GEMM for DeepSeek-V3 Attention Output - Final
 *
 * Dimensions: N=7168, K=1536, M=variable (1,2,3,4,5,8,512)
 * Format: Q4_1 weights (4-bit packed, asymmetric quantization with min)
 * Pattern: llama.cpp BLOCK_Q4_1 x Q8_1 style
 *
 * Hardware: RTX 4090 (CC 8.9, 128 SMs, 1008 GB/s BW)
 *
 * Final Optimizations:
 * - 32 threads per block for maximum occupancy
 * - Launch many blocks to utilize all 128 SMs
 * - Direct FP32 for M<=4, DP4A for M>4
 * - Strategy dispatch based on M
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Constants
constexpr int QK = 32;
constexpr int Q4_1_BLOCK_SIZE = 20;
constexpr int NUM_K_BLOCKS = 48;  // K=1536 / 32 = 48

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

// ============================================================================
// KERNEL 1: Direct FP32 for memory-bound regime (M <= 4)
// 32 threads per block for maximum occupancy
// ============================================================================
__global__ void __launch_bounds__(32) gemm_kernel_direct(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const float* act_row = activation + static_cast<int64_t>(m) * K;
    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; ++kb) {
        const uint8_t* wb = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK_SIZE;

        // Load scale and min
        half2 scale_min = *reinterpret_cast<const half2*>(wb);
        float d_w = __half2float(scale_min.x);
        float m_w = __half2float(scale_min.y);

        const int k_start = kb * QK;
        const uint8_t* qs = wb + 4;

        // Load activations with float4
        float a[QK];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = *reinterpret_cast<const float4*>(&act_row[k_start + i * 4]);
            a[i*4 + 0] = v.x;
            a[i*4 + 1] = v.y;
            a[i*4 + 2] = v.z;
            a[i*4 + 3] = v.w;
        }

        // Direct FP32 computation
        float block_sum_q = 0.0f;
        float block_sum_a = 0.0f;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = qs[i];
            float w0 = static_cast<float>(packed & 0x0F);
            float w1 = static_cast<float>((packed >> 4) & 0x0F);

            block_sum_q += a[i] * w0 + a[i + 16] * w1;
            block_sum_a += a[i] + a[i + 16];
        }

        sum += d_w * block_sum_q + m_w * block_sum_a;
    }

    output[m * N + n] = sum;
}

// ============================================================================
// KERNEL 2: DP4A for larger M
// ============================================================================
__global__ void __launch_bounds__(256) gemm_kernel_dp4a(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const float* act_row = activation + static_cast<int64_t>(m) * K;
    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; ++kb) {
        const uint8_t* wb = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK_SIZE;

        half2 scale_min = *reinterpret_cast<const half2*>(wb);
        float d_w = __half2float(scale_min.x);
        float m_w = __half2float(scale_min.y);

        const int k_start = kb * QK;
        const uint8_t* qs = wb + 4;

        // Load activations using float4
        float a[QK];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = *reinterpret_cast<const float4*>(&act_row[k_start + i * 4]);
            a[i*4 + 0] = v.x;
            a[i*4 + 1] = v.y;
            a[i*4 + 2] = v.z;
            a[i*4 + 3] = v.w;
        }

        // Q8_1 quantization
        float amax = 0.0f;
        float asum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            amax = fmaxf(amax, fabsf(a[i]));
            asum += a[i];
        }
        float d_a = (amax > 1e-10f) ? (amax / 127.0f) : 1.0f;

        // Unpack Q4_1 to INT8
        int8_t w_qs[QK];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            w_qs[i] = static_cast<int8_t>(packed & 0x0F);
            w_qs[i+16] = static_cast<int8_t>((packed >> 4) & 0x0F);
        }

        // Quantize activation
        int8_t a_qs[QK];
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            int q = static_cast<int>(__float2int_rn(a[i] / d_a));
            a_qs[i] = static_cast<int8_t>((q < -128) ? -128 : ((q > 127) ? 127 : q));
        }

        // DP4A dot product
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < QK; i += 4) {
            int a_packed = *reinterpret_cast<int*>(&a_qs[i]);
            int w_packed = *reinterpret_cast<int*>(&w_qs[i]);
            sumi = dp4a(a_packed, w_packed, sumi);
        }

        sum += d_w * d_a * static_cast<float>(sumi) + m_w * asum;
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Host dispatch
// ============================================================================

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 4) {
        // Memory-bound: use 32 threads per block for maximum occupancy
        // N=7168, with 32 threads per block = 224 blocks
        // This allows better load balancing across 128 SMs
        dim3 block(32);
        dim3 grid((N + block.x - 1) / block.x, M);
        gemm_kernel_direct<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Larger M: use DP4A with 256 threads
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x, M);
        gemm_kernel_dp4a<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM Final - Maximum Occupancy");
}
