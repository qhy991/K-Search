#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// Q4_0 block structure (packed, no alignment requirement)
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
 * W4A32C8 Q4_0 Quantized GEMM Kernel - Memory-Optimized Version
 *
 * Optimized for small batch sizes (M=1-8) where the kernel is memory-bound.
 * Uses simple per-thread computation with vectorized loads and efficient
 * memory access patterns.
 *
 * Problem: DeepSeek-V3 LM Head
 * - N = 129280 (output features)
 * - K = 7168 (input features)
 * - M = variable (batch size, 1-512)
 *
 * Computation: C = A @ W^T where:
 * - A: FP32 activations [M, K]
 * - W: Q4_0 quantized weights [N, K/32] (block_q4_0 format)
 * - C: FP32 output [M, N]
 *
 * Formula: result = d4_0 * (d_a * sumi - 8 * s_a)
 * where the -8*s_a term compensates for Q4_0's offset-8 encoding.
 */
__global__ void w4a32c8_q4_0_gemm_kernel(
    const uint8_t* __restrict__ weight,      // Q4_0 weights [N, K/32, 18]
    const float* __restrict__ activation,    // FP32 activations [M, K]
    float* __restrict__ output,              // FP32 output [M, N]
    int M, int N, int K
) {
    // Each thread computes one output element
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    // Each weight row has K/32 blocks
    const int num_blocks = K / 32;

    // Pointer to the n-th weight row (each row has num_blocks * 18 bytes)
    const block_q4_0* w_row = reinterpret_cast<const block_q4_0*>(weight + n * num_blocks * 18);

    // Pointer to the m-th activation row
    const float* a_row = activation + m * K;

    float acc = 0.0f;

    // Iterate over blocks
    for (int b = 0; b < num_blocks; b++) {
        // Load Q4_0 weight block
        const block_q4_0 w_block = w_row[b];
        float d_w = read_half_as_float(w_block.d);

        // Pointer to activation block start (32 values)
        const float* a_block = a_row + b * 32;

        // Dynamically quantize activation block to Q8_1 style
        // Find max absolute value for scaling
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
        }

        // Compute activation scale (d_a)
        float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

        // Compute activation sum (s_a) - sum of FP32 values
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_sum += a_block[i];
        }

        // Compute INT8 dot product with Q4_0 unpacking
        // Q4_0 format: qs[16] contains 32 packed 4-bit values
        // Each byte has low nibble (value i) and high nibble (value i+16)
        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = w_block.qs[i];
            int w_low = byte_val & 0x0F;           // value in [0, 15]
            int w_high = (byte_val >> 4) & 0x0F;   // value in [0, 15]

            // Quantize activation values to int8
            int a_low = __float2int_rn(a_block[i] / d_a);
            int a_high = __float2int_rn(a_block[i + 16] / d_a);

            sumi += w_low * a_low;
            sumi += w_high * a_high;
        }

        // Apply llama.cpp formula: d4_0 * (d_a * sumi - 8 * s_a)
        // The -8*s_a term compensates for Q4_0's offset-8 encoding
        acc += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
    }

    output[m * N + n] = acc;
}

/**
 * Host function to launch the kernel
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

    // Launch configuration optimized for memory bandwidth
    // Use larger thread blocks for better coalescing
    const int threads_per_block = 256;
    const int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    const int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block, 1);

    // Launch kernel
    w4a32c8_q4_0_gemm_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM (DeepSeek-V3 LM Head)");
}
