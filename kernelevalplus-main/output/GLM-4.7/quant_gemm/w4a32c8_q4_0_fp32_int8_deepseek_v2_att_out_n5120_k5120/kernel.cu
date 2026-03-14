#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 block structure (llama.cpp compatible)
struct block_q4_0 {
    uint16_t d;      // FP16 scale
    uint8_t qs[16];  // 32 packed quaternions
};

// Convert FP16 to FP32 using union for safety
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Simple kernel for small M (one thread per output element)
__global__ void q4_0_fp32_gemm_small_m_kernel(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0& wb = weight[n * num_blocks_k + kb];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = kb * 32;

        // Unpack and compute in single pass
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = wb.qs[i];
            int q0 = packed & 0x0F;
            int q1 = (packed >> 4) & 0x0F;

            float a0 = activation[m * K + k_base + i];
            float a1 = activation[m * K + k_base + i + 16];

            sum += a0 * d_w * (q0 - 8);
            sum += a1 * d_w * (q1 - 8);
        }
    }

    output[m * N + n] = sum;
}

// Optimized kernel for large M (8x8 thread blocks)
__global__ void __launch_bounds__(64) q4_0_fp32_gemm_large_m_kernel(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int tx = threadIdx.x;  // 0-7 (M dimension)
    const int ty = threadIdx.y;  // 0-7 (N dimension)

    const int m = blockIdx.y * 8 + tx;
    const int n = blockIdx.x * 8 + ty;

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int weight_row_base = n * num_blocks_k;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0& wb = weight[weight_row_base + kb];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = kb * 32;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = wb.qs[i];
            int q0 = packed & 0x0F;
            int q1 = (packed >> 4) & 0x0F;

            float a0 = activation[m * K + k_base + i];
            float a1 = activation[m * K + k_base + i + 16];

            sum += a0 * d_w * (q0 - 8);
            sum += a1 * d_w * (q1 - 8);
        }
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Strategy dispatch: choose kernel based on M
    if (M <= 8) {
        // Small M: one thread per output element
        dim3 grid(N, M);
        dim3 block(1);
        q4_0_fp32_gemm_small_m_kernel<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large M: 8x8 thread blocks for better parallelism
        dim3 grid((N + 7) / 8, (M + 7) / 8);
        dim3 block(8, 8);
        q4_0_fp32_gemm_large_m_kernel<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 FP32 GEMM with strategy dispatch");
}
