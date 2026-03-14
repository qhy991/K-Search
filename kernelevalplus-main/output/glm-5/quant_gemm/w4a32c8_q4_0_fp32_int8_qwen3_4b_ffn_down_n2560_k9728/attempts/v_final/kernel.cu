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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (Final)
 *
 * Qwen3-4B FFN Down projection
 * - N = 2560 (output features)
 * - K = 9728 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * This is the final optimized version based on v3 (best overall performance).
 * Uses simple FP32 dequantization approach for correctness and performance.
 */

#define Q4_0_BLOCK_SIZE 32

// Device function to convert uint16_t to float (FP16 half)
__device__ __inline__ float half_to_float(const uint8_t* p) {
    uint16_t u = p[0] | ((uint16_t)p[1] << 8);
    union { uint16_t u16; __half f16; } un;
    un.u16 = u;
    return __half2float(un.f16);
}

// Kernel implementation - one thread per output element
// Uses Q4_0 dequantization with FP32 activation
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

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_idx * num_blocks * 18;

    // Iterate through all K blocks
    // Use loop unrolling for better instruction-level parallelism
    int kb = 0;
    for (; kb + 3 < num_blocks; kb += 4) {
        // Process 4 blocks at a time for better ILP
        const float* ab0 = act_row + kb * 32;
        const float* ab1 = ab0 + 32;
        const float* ab2 = ab1 + 32;
        const float* ab3 = ab2 + 32;

        const uint8_t* wb0 = weight_row + kb * 18;
        const uint8_t* wb1 = wb0 + 18;
        const uint8_t* wb2 = wb1 + 18;
        const uint8_t* wb3 = wb2 + 18;

        // Read scales (FP16 -> FP32)
        float dw0 = half_to_float(wb0);
        float dw1 = half_to_float(wb1);
        float dw2 = half_to_float(wb2);
        float dw3 = half_to_float(wb3);

        const uint8_t* wqs0 = wb0 + 2;
        const uint8_t* wqs1 = wb1 + 2;
        const uint8_t* wqs2 = wb2 + 2;
        const uint8_t* wqs3 = wb3 + 2;

        // Unroll for better throughput
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b0 = wqs0[i];
            uint8_t b1 = wqs1[i];
            uint8_t b2 = wqs2[i];
            uint8_t b3 = wqs3[i];

            // Q4_0 uses offset-8 encoding
            acc += ab0[i] * (dw0 * ((int)(b0 & 0x0F) - 8));
            acc += ab0[i + 16] * (dw0 * ((int)((b0 >> 4) & 0x0F) - 8));
            acc += ab1[i] * (dw1 * ((int)(b1 & 0x0F) - 8));
            acc += ab1[i + 16] * (dw1 * ((int)((b1 >> 4) & 0x0F) - 8));
            acc += ab2[i] * (dw2 * ((int)(b2 & 0x0F) - 8));
            acc += ab2[i + 16] * (dw2 * ((int)((b2 >> 4) & 0x0F) - 8));
            acc += ab3[i] * (dw3 * ((int)(b3 & 0x0F) - 8));
            acc += ab3[i + 16] * (dw3 * ((int)((b3 >> 4) & 0x0F) - 8));
        }
    }

    // Handle remaining blocks
    for (; kb < num_blocks; kb++) {
        const float* act_block = act_row + kb * 32;
        const uint8_t* weight_block = weight_row + kb * 18;

        float dw = half_to_float(weight_block);
        const uint8_t* w_qs = weight_block + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = w_qs[i];
            acc += act_block[i] * (dw * ((int)(b & 0x0F) - 8));
            acc += act_block[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
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
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (Final)");
}
