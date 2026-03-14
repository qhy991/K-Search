#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v3 - Simplified & Fixed)
 *
 * Qwen3-4B FFN Down projection
 * - N = 2560 (output features)
 * - K = 9728 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Q4_0 format (18 bytes per block):
 * - Bytes 0-1: half precision scale factor d
 * - Bytes 2-17: 16 uint8 values, each containing 2 packed 4-bit values
 *
 * For W4A32C8, activation is FP32 (not quantized).
 * Computation: output = activation @ (d_w * (q_w - 8))^T
 *
 * This version:
 * 1. Simplifies the indexing for correctness
 * 2. Uses efficient memory patterns
 * 3. Properly handles the Q4_0 unpacking
 */

#define Q4_0_BLOCK_SIZE 32

// Device function to read FP16 as float
__device__ __inline__ float half_to_float(const uint8_t* p) {
    uint16_t u = p[0] | ((uint16_t)p[1] << 8);
    union { uint16_t u16; __half f16; } un;
    un.u16 = u;
    return __half2float(un.f16);
}

// Simple kernel for correctness first - one thread per output element
__global__ void quant_gemm_q4_0_fp32_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    float acc = 0.0f;

    // Each row of activation is size K
    const float* act_row = activation + m_idx * K;

    // Each row of weight has num_blocks * 18 bytes
    const uint8_t* weight_row_base = weight + n_idx * num_blocks * 18;

    // Iterate through all K blocks
    for (int kb = 0; kb < num_blocks; kb++) {
        // Get pointers to current block
        const float* act_block = act_row + kb * 32;
        const uint8_t* weight_block = weight_row_base + kb * 18;

        // Read scale (FP16 -> FP32)
        float dw = half_to_float(weight_block);

        // Q4_0: 16 bytes of packed 4-bit values
        const uint8_t* w_packed = weight_block + 2;

        // Unroll for performance
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_packed[i];

            // Low nibble corresponds to position i (0-15)
            // High nibble corresponds to position i+16 (16-31)
            int w_low = (packed & 0x0F) - 8;   // Q4_0 offset-8 encoding
            int w_high = ((packed >> 4) & 0x0F) - 8;

            acc += act_block[i] * (dw * w_low);
            acc += act_block[i + 16] * (dw * w_high);
        }
    }

    output[m_idx * N + n_idx] = acc;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    auto weight_contig = weight.contiguous();
    auto act_contig = activation.contiguous();

    // Use warp-aligned block size for better performance
    int threads_per_block = 256;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    quant_gemm_q4_0_fp32_kernel<<<grid, block>>>(
        (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
        act_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V3 - Simplified)");
}
