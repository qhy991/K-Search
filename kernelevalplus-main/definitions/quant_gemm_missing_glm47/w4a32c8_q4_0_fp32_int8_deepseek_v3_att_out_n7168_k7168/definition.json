#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure (llama.cpp compatible)
struct block_q4_0 {
    uint16_t d;      // FP16 scale
    uint8_t qs[16];  // 32 packed quaternions
};

__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Simple kernel for small M (one thread per output element)
// Direct FP32 computation - much faster than activation quantization
__global__ void q4_0_gemm_small_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Thread ID mapped to output position
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = tid / N;
    const int n = tid % N;

    if (m >= M || n >= N) return;

    float sum = 0.0f;

    const float* a_row = activation + m * K;
    const block_q4_0* w_col = weight_blocks + n * num_blocks;

    // Process each block
    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_0& wb = w_col[b];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = b * 32;

        // Unpack and compute in single pass - no activation quantization
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = wb.qs[i];
            int q0 = packed & 0x0F;
            int q1 = (packed >> 4) & 0x0F;

            float a0 = a_row[k_base + i];
            float a1 = a_row[k_base + i + 16];

            sum += a0 * d_w * (q0 - 8);
            sum += a1 * d_w * (q1 - 8);
        }
    }

    output[m * N + n] = sum;
}

// Optimized kernel for large M (8x8 thread blocks)
__global__ void q4_0_gemm_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Thread indices
    const int tx = threadIdx.x;  // 0-7 (N dimension)
    const int ty = threadIdx.y;  // 0-7 (M dimension)

    const int m = blockIdx.y * 8 + ty;
    const int n = blockIdx.x * 8 + tx;

    if (m >= M || n >= N) return;

    float sum = 0.0f;

    // Process each block
    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_0& wb = weight_blocks[n * num_blocks + b];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = b * 32;

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

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const float* a_ptr = activation.data_ptr<float>();
    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    float* o_ptr = output.data_ptr<float>();

    // Strategy dispatch based on M
    if (M <= 8) {
        // Small M: simple 1D block layout
        int num_threads = 256;
        int num_blocks = (M * N + num_threads - 1) / num_threads;
        q4_0_gemm_small_m_kernel<<<num_blocks, num_threads>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    } else {
        // Large M: 8x8 2D thread blocks for better parallelism
        dim3 block(8, 8);
        dim3 grid((N + 7) / 8, (M + 7) / 8);
        q4_0_gemm_large_m_kernel<<<grid, block>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
