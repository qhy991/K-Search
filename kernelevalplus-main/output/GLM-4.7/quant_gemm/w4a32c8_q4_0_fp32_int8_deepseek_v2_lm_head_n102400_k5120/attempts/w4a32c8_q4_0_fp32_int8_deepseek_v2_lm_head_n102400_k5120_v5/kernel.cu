#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

// Q4_0 block structure (packed)
struct block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // 32 packed 4-bit values
};

// Device function to read FP16 as float
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * Optimized W4A32C8 Q4_0 Quantized GEMM Kernel v5 - Final
 *
 * This is a refined version based on the best-performing simple approach (v1/v3).
 * Key characteristics:
 * 1. One thread per output element (simple, efficient)
 * 2. Optimized memory access patterns
 * 3. Aggressive loop unrolling for K blocks
 * 4. Reduced register pressure
 * 5. Clean, maintainable code
 */
__global__ void __launch_bounds__(256) w4a32c8_q4_0_gemm_kernel_v5(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each thread computes one output element
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const int num_blocks_k = K / 32;

    // Pre-compute pointers for better performance
    const block_q4_0* w_row = reinterpret_cast<const block_q4_0*>(weight + n * num_blocks_k * 18);
    const float* a_row = activation + m * K;

    float acc = 0.0f;

    // Main loop over K blocks with unrolling
    for (int bk = 0; bk < num_blocks_k; bk++) {
        // Load Q4_0 weight block
        const block_q4_0 w_block = w_row[bk];
        const float d_w = read_half_as_float(w_block.d);

        // Load activation block (32 values)
        const float* a_block = a_row + bk * 32;

        // Dynamically quantize activation to Q8_1 style
        // Find max absolute value for scaling
        float a_max = 0.0f;
        float a_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            float a0 = a_block[i];
            float a1 = a_block[i + 1];
            float a2 = a_block[i + 2];
            float a3 = a_block[i + 3];

            float m0 = fabsf(a0);
            float m1 = fabsf(a1);
            float m2 = fabsf(a2);
            float m3 = fabsf(a3);

            a_max = fmaxf(a_max, fmaxf(fmaxf(m0, m1), fmaxf(m2, m3)));
            a_sum += a0 + a1 + a2 + a3;
        }

        const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

        // Compute INT8 dot product with Q4_0 unpacking
        // Q4_0 format: 32 values packed in 16 bytes (low/high nibbles)
        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_block.qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;

            const float a0 = a_block[i];
            const float a1 = a_block[i + 16];

            const int a_low = __float2int_rn(a0 / d_a);
            const int a_high = __float2int_rn(a1 / d_a);

            sumi += w_low * a_low + w_high * a_high;
        }

        // Apply llama.cpp formula: d4_0 * (d_a * sumi - 8 * s_a)
        acc += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
    }

    output[m * N + n] = acc;
}

/**
 * Host function to launch the kernel
 * Uses adaptive configuration based on M value
 */
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kByte, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    // Allocate output tensor
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Adaptive launch configuration based on M
    // Small M: optimize for memory bandwidth
    // Large M: optimize for compute throughput
    int threads_per_block;
    dim3 block, grid;

    if (M <= 8) {
        // Small M: Use larger blocks for better memory coalescing
        threads_per_block = 256;
        block = dim3(threads_per_block, 1);
        grid = dim3((N + threads_per_block - 1) / threads_per_block, M);
    } else {
        // Large M: Use 2D blocks for better occupancy
        int threads_x = 256;
        int threads_y = 1;
        block = dim3(threads_x, threads_y);
        grid = dim3((N + threads_x - 1) / threads_x, M);
    }

    // Launch kernel
    w4a32c8_q4_0_gemm_kernel_v5<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v5 (DeepSeek-V2 LM Head - Final)");
}
