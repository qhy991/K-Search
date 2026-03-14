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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v7 - Highly Optimized)
 *
 * DeepSeek-V2 MoE Routing Down Projection
 * - N = 1536 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Optimizations v7:
 * - Process 4 K-blocks per iteration for better ILP
 * - Complete loop unrolling
 * - Register tiling for activation values
 * - Fused dequantization and MAC
 */

#define Q4_0_BLOCK_SIZE 32

/**
 * Highly optimized kernel with aggressive unrolling
 */
__global__ void __launch_bounds__(256)
quant_gemm_q4_0_fp32_kernel_v7(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + tid;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;

    // Register file for partial results (process 4 blocks at once)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_idx * num_blocks * 18;

    // Process 4 blocks per iteration
    int kb = 0;
    const int num_blocks_4 = (num_blocks / 4) * 4;

    for (; kb < num_blocks_4; kb += 4) {
        // Load 4 weight scales
        half d_w0, d_w1, d_w2, d_w3;
        const uint8_t* wb0 = weight_row + kb * 18;
        const uint8_t* wb1 = wb0 + 18;
        const uint8_t* wb2 = wb1 + 18;
        const uint8_t* wb3 = wb2 + 18;
        memcpy(&d_w0, wb0, sizeof(half));
        memcpy(&d_w1, wb1, sizeof(half));
        memcpy(&d_w2, wb2, sizeof(half));
        memcpy(&d_w3, wb3, sizeof(half));
        float dw0 = __half2float(d_w0);
        float dw1 = __half2float(d_w1);
        float dw2 = __half2float(d_w2);
        float dw3 = __half2float(d_w3);

        const uint8_t* wqs0 = wb0 + 2;
        const uint8_t* wqs1 = wb1 + 2;
        const uint8_t* wqs2 = wb2 + 2;
        const uint8_t* wqs3 = wb3 + 2;

        const float* ab0 = act_row + kb * 32;
        const float* ab1 = ab0 + 32;
        const float* ab2 = ab1 + 32;
        const float* ab3 = ab2 + 32;

        // Fully unrolled compute for all 4 blocks
        // Each iteration: load 4 packed bytes, dequantize, MAC
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Block 0
            uint8_t b0 = wqs0[i];
            int w0_l = (b0 & 0x0F) - 8, w0_h = ((b0 >> 4) & 0x0F) - 8;
            acc0 += ab0[i] * (dw0 * w0_l);
            acc0 += ab0[i + 16] * (dw0 * w0_h);

            // Block 1
            uint8_t b1 = wqs1[i];
            int w1_l = (b1 & 0x0F) - 8, w1_h = ((b1 >> 4) & 0x0F) - 8;
            acc1 += ab1[i] * (dw1 * w1_l);
            acc1 += ab1[i + 16] * (dw1 * w1_h);

            // Block 2
            uint8_t b2 = wqs2[i];
            int w2_l = (b2 & 0x0F) - 8, w2_h = ((b2 >> 4) & 0x0F) - 8;
            acc2 += ab2[i] * (dw2 * w2_l);
            acc2 += ab2[i + 16] * (dw2 * w2_h);

            // Block 3
            uint8_t b3 = wqs3[i];
            int w3_l = (b3 & 0x0F) - 8, w3_h = ((b3 >> 4) & 0x0F) - 8;
            acc3 += ab3[i] * (dw3 * w3_l);
            acc3 += ab3[i + 16] * (dw3 * w3_h);
        }

        #pragma unroll
        for (int i = 4; i < 8; i++) {
            uint8_t b0 = wqs0[i];
            int w0_l = (b0 & 0x0F) - 8, w0_h = ((b0 >> 4) & 0x0F) - 8;
            acc0 += ab0[i] * (dw0 * w0_l);
            acc0 += ab0[i + 16] * (dw0 * w0_h);

            uint8_t b1 = wqs1[i];
            int w1_l = (b1 & 0x0F) - 8, w1_h = ((b1 >> 4) & 0x0F) - 8;
            acc1 += ab1[i] * (dw1 * w1_l);
            acc1 += ab1[i + 16] * (dw1 * w1_h);

            uint8_t b2 = wqs2[i];
            int w2_l = (b2 & 0x0F) - 8, w2_h = ((b2 >> 4) & 0x0F) - 8;
            acc2 += ab2[i] * (dw2 * w2_l);
            acc2 += ab2[i + 16] * (dw2 * w2_h);

            uint8_t b3 = wqs3[i];
            int w3_l = (b3 & 0x0F) - 8, w3_h = ((b3 >> 4) & 0x0F) - 8;
            acc3 += ab3[i] * (dw3 * w3_l);
            acc3 += ab3[i + 16] * (dw3 * w3_h);
        }

        #pragma unroll
        for (int i = 8; i < 12; i++) {
            uint8_t b0 = wqs0[i];
            int w0_l = (b0 & 0x0F) - 8, w0_h = ((b0 >> 4) & 0x0F) - 8;
            acc0 += ab0[i] * (dw0 * w0_l);
            acc0 += ab0[i + 16] * (dw0 * w0_h);

            uint8_t b1 = wqs1[i];
            int w1_l = (b1 & 0x0F) - 8, w1_h = ((b1 >> 4) & 0x0F) - 8;
            acc1 += ab1[i] * (dw1 * w1_l);
            acc1 += ab1[i + 16] * (dw1 * w1_h);

            uint8_t b2 = wqs2[i];
            int w2_l = (b2 & 0x0F) - 8, w2_h = ((b2 >> 4) & 0x0F) - 8;
            acc2 += ab2[i] * (dw2 * w2_l);
            acc2 += ab2[i + 16] * (dw2 * w2_h);

            uint8_t b3 = wqs3[i];
            int w3_l = (b3 & 0x0F) - 8, w3_h = ((b3 >> 4) & 0x0F) - 8;
            acc3 += ab3[i] * (dw3 * w3_l);
            acc3 += ab3[i + 16] * (dw3 * w3_h);
        }

        #pragma unroll
        for (int i = 12; i < 16; i++) {
            uint8_t b0 = wqs0[i];
            int w0_l = (b0 & 0x0F) - 8, w0_h = ((b0 >> 4) & 0x0F) - 8;
            acc0 += ab0[i] * (dw0 * w0_l);
            acc0 += ab0[i + 16] * (dw0 * w0_h);

            uint8_t b1 = wqs1[i];
            int w1_l = (b1 & 0x0F) - 8, w1_h = ((b1 >> 4) & 0x0F) - 8;
            acc1 += ab1[i] * (dw1 * w1_l);
            acc1 += ab1[i + 16] * (dw1 * w1_h);

            uint8_t b2 = wqs2[i];
            int w2_l = (b2 & 0x0F) - 8, w2_h = ((b2 >> 4) & 0x0F) - 8;
            acc2 += ab2[i] * (dw2 * w2_l);
            acc2 += ab2[i + 16] * (dw2 * w2_h);

            uint8_t b3 = wqs3[i];
            int w3_l = (b3 & 0x0F) - 8, w3_h = ((b3 >> 4) & 0x0F) - 8;
            acc3 += ab3[i] * (dw3 * w3_l);
            acc3 += ab3[i + 16] * (dw3 * w3_h);
        }
    }

    // Process remaining blocks
    for (; kb < num_blocks; kb++) {
        const float* act_block = act_row + kb * 32;
        const uint8_t* weight_block = weight_row + kb * 18;

        half d_w;
        memcpy(&d_w, weight_block, sizeof(half));
        float dw = __half2float(d_w);
        const uint8_t* w_qs = weight_block + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = w_qs[i];
            int w_l = (b & 0x0F) - 8, w_h = ((b >> 4) & 0x0F) - 8;
            acc0 += act_block[i] * (dw * w_l);
            acc0 += act_block[i + 16] * (dw * w_h);
        }
    }

    output[m_idx * N + n_idx] = acc0 + acc1 + acc2 + acc3;
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

    int threads_per_block = 256;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    quant_gemm_q4_0_fp32_kernel_v7<<<grid, block>>>(
        (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
        act_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V7 Optimized)");
}
