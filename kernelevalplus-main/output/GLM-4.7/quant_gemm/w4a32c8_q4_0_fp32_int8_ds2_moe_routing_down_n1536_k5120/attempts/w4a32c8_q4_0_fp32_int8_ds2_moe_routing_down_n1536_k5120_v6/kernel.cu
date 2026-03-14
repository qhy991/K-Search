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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v6 - DP4A Optimized)
 *
 * DeepSeek-V2 MoE Routing Down Projection
 * - N = 1536 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Optimizations v6:
 * - Uses DP4A (dot product accumulate 4-element) instruction
 * - Vectorized loads with float4
 * - Fused dequantization and computation
 * - Better thread block configuration for compute-bound workload
 */

#define Q4_0_BLOCK_SIZE 32

// DP4A helper: computes dot product of 4 int8 pairs
// result = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
__device__ __forceinline__ int dp4a(int a, int b) {
    int result;
    asm volatile("dp4a.s32.s32 %0, %1, %2, 0;"
                 : "=r"(result)
                 : "r"(a), "r"(b));
    return result;
}

/**
 * Optimized kernel using DP4A for INT8 dot products
 * Each thread computes one output element
 */
__global__ void __launch_bounds__(256)
quant_gemm_q4_0_fp32_kernel_dp4a(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Thread layout
    const int tid = threadIdx.x;
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + tid;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    float acc = 0.0f;

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_idx * num_blocks * 18;

    // Process multiple blocks at once for better ILP
    int kb = 0;

    // Main loop: process 2 blocks per iteration
    for (; kb + 1 < num_blocks; kb += 2) {
        // Load activation blocks (keep in registers)
        float a0[32], a1[32];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v0 = *((float4*)(act_row + kb * 32 + i * 4));
            float4 v1 = *((float4*)(act_row + (kb + 1) * 32 + i * 4));
            a0[i * 4 + 0] = v0.x; a0[i * 4 + 1] = v0.y; a0[i * 4 + 2] = v0.z; a0[i * 4 + 3] = v0.w;
            a1[i * 4 + 0] = v1.x; a1[i * 4 + 1] = v1.y; a1[i * 4 + 2] = v1.z; a1[i * 4 + 3] = v1.w;
        }

        // Load weight scales
        half d_w0, d_w1;
        const uint8_t* wb0 = weight_row + kb * 18;
        const uint8_t* wb1 = weight_row + (kb + 1) * 18;
        memcpy(&d_w0, wb0, sizeof(half));
        memcpy(&d_w1, wb1, sizeof(half));
        float dw0_f = __half2float(d_w0);
        float dw1_f = __half2float(d_w1);

        const uint8_t* wqs0 = wb0 + 2;
        const uint8_t* wqs1 = wb1 + 2;

        // Compute block 0
        {
            // Quantize activations to int8 on the fly
            int8_t aq0[32];
            float max0 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                max0 = fmaxf(max0, fabsf(a0[i]));
            }
            float scale0 = max0 / 127.0f;
            if (scale0 < 1e-7f) scale0 = 1e-7f;

            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int v = __float2int_rn(a0[i] / scale0);
                aq0[i] = (int8_t)max(-128, min(127, v));
            }

            // Unpack Q4_0 and compute with DP4A
            int sum0 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int w_packed0 = ((const int*)wqs0)[i];
                int a_packed0 = *((int*)&aq0[i * 4]);
                sum0 += dp4a(w_packed0, a_packed0);
            }

            // Dequantize contribution: dw0 * (da0 * sum0 - 8 * sum_a0)
            float sum_a0 = 0;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sum_a0 += a0[i];
            }
            acc += dw0_f * (scale0 * (float)sum0 - 8.0f * sum_a0);
        }

        // Compute block 1
        {
            int8_t aq1[32];
            float max1 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                max1 = fmaxf(max1, fabsf(a1[i]));
            }
            float scale1 = max1 / 127.0f;
            if (scale1 < 1e-7f) scale1 = 1e-7f;

            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int v = __float2int_rn(a1[i] / scale1);
                aq1[i] = (int8_t)max(-128, min(127, v));
            }

            int sum1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int w_packed1 = ((const int*)wqs1)[i];
                int a_packed1 = *((int*)&aq1[i * 4]);
                sum1 += dp4a(w_packed1, a_packed1);
            }

            float sum_a1 = 0;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sum_a1 += a1[i];
            }
            acc += dw1_f * (scale1 * (float)sum1 - 8.0f * sum_a1);
        }
    }

    // Handle remaining block
    if (kb < num_blocks) {
        const float* act_block = act_row + kb * 32;
        const uint8_t* weight_block = weight_row + kb * 18;

        half d_w;
        memcpy(&d_w, weight_block, sizeof(half));
        float d_w_f = __half2float(d_w);

        const uint8_t* w_qs = weight_block + 2;

        // Direct dequantization and compute (faster for single block)
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = w_qs[i];
            int w_low = byte_val & 0x0F;
            int w_high = (byte_val >> 4) & 0x0F;

            float w_low_f = d_w_f * (float)(w_low - 8);
            float w_high_f = d_w_f * (float)(w_high - 8);

            acc += act_block[i] * w_low_f;
            acc += act_block[i + 16] * w_high_f;
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

    int threads_per_block = 256;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    quant_gemm_q4_0_fp32_kernel_dp4a<<<grid, block>>>(
        (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
        act_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V6 DP4A)");
}
