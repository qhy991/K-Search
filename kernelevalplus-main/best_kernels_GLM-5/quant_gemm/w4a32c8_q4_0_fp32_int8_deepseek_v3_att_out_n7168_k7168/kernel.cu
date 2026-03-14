/**
 * W4A32C8 Q4_0 Quantized GEMM for DeepSeek-V3 Attention Output - V10
 *
 * Strategy: Pre-dequantize weights to FP32 and use cuBLAS for GEMM.
 *
 * This approach trades memory for compute efficiency:
 * - Dequantization overhead: O(N*K) memory and compute
 * - GEMM efficiency: cuBLAS is highly optimized
 *
 * For repeated calls with the same weights, dequantization is amortized.
 * For single calls, this tests the upper bound of achievable performance.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Constants
// ============================================================================
constexpr int QK = 32;
constexpr int Q4_0_BYTES = 18;

// ============================================================================
// Q4_0 Block Structure
// ============================================================================
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(block_q4_0) == 18, "Q4_0 block must be 18 bytes");

// ============================================================================
// KERNEL: Dequantize Q4_0 to FP32
// ============================================================================

__global__ void dequantize_q4_0_kernel(
    const uint8_t* __restrict__ weight,
    float* __restrict__ weight_fp32,
    int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_blocks = K / QK;

    if (n >= N) return;

    for (int kb = 0; kb < num_blocks; ++kb) {
        const block_q4_0* wb = reinterpret_cast<const block_q4_0*>(
            weight + (n * num_blocks + kb) * Q4_0_BYTES
        );

        const float d = __half2float(wb->d);
        const int k_start = kb * QK;

        // Low nibbles -> positions 0-15
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int nibble = wb->qs[i] & 0x0F;
            weight_fp32[n * K + k_start + i] = (nibble - 8) * d;
        }

        // High nibbles -> positions 16-31
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int nibble = (wb->qs[i] >> 4) & 0x0F;
            weight_fp32[n * K + k_start + i + 16] = (nibble - 8) * d;
        }
    }
}

// ============================================================================
// KERNEL: Direct FP32 GEMV (without cuBLAS)
// ============================================================================

__global__ void __launch_bounds__(256) gemv_fp32_kernel(
    const float* __restrict__ weight_t,  // K x N, transposed
    const float* __restrict__ activation, // K elements
    float* __restrict__ output,           // N elements
    int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N) return;

    float sum = 0.0f;

    // Vectorized dot product
    const float* w_row = weight_t + n * K;

    #pragma unroll 8
    for (int k = 0; k < K; k += 4) {
        float4 w4 = *reinterpret_cast<const float4*>(w_row + k);
        float4 a4 = *reinterpret_cast<const float4*>(activation + k);
        sum += w4.x * a4.x;
        sum += w4.y * a4.y;
        sum += w4.z * a4.z;
        sum += w4.w * a4.w;
    }

    output[n] = sum;
}

// ============================================================================
// KERNEL: Optimized Q4_0 GEMV with inline dequantization
// ============================================================================

__global__ void __launch_bounds__(256) gemv_q4_0_inline_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N) return;

    const int num_blocks = K / QK;

    float sum = 0.0f;

    // Process K in chunks of 32
    for (int kb = 0; kb < num_blocks; ++kb) {
        const block_q4_0* wb = reinterpret_cast<const block_q4_0*>(
            weight + (n * num_blocks + kb) * Q4_0_BYTES
        );

        const float d = __half2float(wb->d);
        const int k_start = kb * QK;

        // Load activation (vectorized)
        float a0 = activation[k_start + 0];
        float a1 = activation[k_start + 1];
        float a2 = activation[k_start + 2];
        float a3 = activation[k_start + 3];
        float a4 = activation[k_start + 4];
        float a5 = activation[k_start + 5];
        float a6 = activation[k_start + 6];
        float a7 = activation[k_start + 7];
        float a8 = activation[k_start + 8];
        float a9 = activation[k_start + 9];
        float a10 = activation[k_start + 10];
        float a11 = activation[k_start + 11];
        float a12 = activation[k_start + 12];
        float a13 = activation[k_start + 13];
        float a14 = activation[k_start + 14];
        float a15 = activation[k_start + 15];
        float a16 = activation[k_start + 16];
        float a17 = activation[k_start + 17];
        float a18 = activation[k_start + 18];
        float a19 = activation[k_start + 19];
        float a20 = activation[k_start + 20];
        float a21 = activation[k_start + 21];
        float a22 = activation[k_start + 22];
        float a23 = activation[k_start + 23];
        float a24 = activation[k_start + 24];
        float a25 = activation[k_start + 25];
        float a26 = activation[k_start + 26];
        float a27 = activation[k_start + 27];
        float a28 = activation[k_start + 28];
        float a29 = activation[k_start + 29];
        float a30 = activation[k_start + 30];
        float a31 = activation[k_start + 31];

        // Process all 16 weight bytes
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = wb->qs[i];
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4) - 8;
            float w_lo = lo * d;
            float w_hi = hi * d;

            // Positions i and i+16
            sum += (i == 0 ? a0 : i == 1 ? a1 : i == 2 ? a2 : i == 3 ? a3 :
                    i == 4 ? a4 : i == 5 ? a5 : i == 6 ? a6 : i == 7 ? a7 :
                    i == 8 ? a8 : i == 9 ? a9 : i == 10 ? a10 : i == 11 ? a11 :
                    i == 12 ? a12 : i == 13 ? a13 : i == 14 ? a14 : a15) * w_lo;

            sum += (i == 0 ? a16 : i == 1 ? a17 : i == 2 ? a18 : i == 3 ? a19 :
                    i == 4 ? a20 : i == 5 ? a21 : i == 6 ? a22 : i == 7 ? a23 :
                    i == 8 ? a24 : i == 9 ? a25 : i == 10 ? a26 : i == 11 ? a27 :
                    i == 12 ? a28 : i == 13 ? a29 : i == 14 ? a30 : a31) * w_hi;
        }
    }

    output[n] = sum;
}

// ============================================================================
// Host Dispatch Function
// ============================================================================

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        // GEMV - use inline dequantization kernel
        const int threads = 256;
        dim3 block(threads);
        dim3 grid((N + threads - 1) / threads);

        gemv_q4_0_inline_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );
    } else {
        // For M > 1, dequantize weights and use matrix multiply
        auto weight_fp32 = torch::empty({N, K},
            torch::dtype(torch::kFloat32).device(weight.device()));

        const int threads = 256;
        dim3 block(threads);
        dim3 grid((N + threads - 1) / threads);

        dequantize_q4_0_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            weight_fp32.data_ptr<float>(),
            N, K
        );

        // Use PyTorch's matmul (cuBLAS)
        output = torch::matmul(activation, weight_fp32.t());
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 GEMM DeepSeek-V3 AttOut V10");
}
