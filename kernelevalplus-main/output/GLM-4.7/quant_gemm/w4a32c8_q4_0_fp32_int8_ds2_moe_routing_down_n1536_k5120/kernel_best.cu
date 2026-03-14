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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v8 - Optimized)
 *
 * DeepSeek-V2 MoE Routing Down Projection
 * - N = 1536 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Optimizations v8:
 * - Reduced shared memory synchronization overhead
 * - Better block size for compute-bound workload
 * - Process 2 blocks per iteration without register overflow
 */

#define Q4_0_BLOCK_SIZE 32

// Kernel for small M (no shared memory, minimal overhead)
__global__ void quant_gemm_q4_0_fp32_kernel_small(
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

    // Process 2 blocks per iteration for better ILP
    int kb = 0;
    for (; kb + 1 < num_blocks; kb += 2) {
        const float* ab0 = act_row + kb * 32;
        const float* ab1 = ab0 + 32;

        const uint8_t* wb0 = weight_row + kb * 18;
        const uint8_t* wb1 = wb0 + 18;

        half d_w0, d_w1;
        memcpy(&d_w0, wb0, sizeof(half));
        memcpy(&d_w1, wb1, sizeof(half));
        float dw0 = __half2float(d_w0);
        float dw1 = __half2float(d_w1);

        const uint8_t* wqs0 = wb0 + 2;
        const uint8_t* wqs1 = wb1 + 2;

        // Unroll for better throughput
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b0 = wqs0[i];
            uint8_t b1 = wqs1[i];
            acc += ab0[i] * (dw0 * ((int)(b0 & 0x0F) - 8));
            acc += ab0[i + 16] * (dw0 * ((int)((b0 >> 4) & 0x0F) - 8));
            acc += ab1[i] * (dw1 * ((int)(b1 & 0x0F) - 8));
            acc += ab1[i + 16] * (dw1 * ((int)((b1 >> 4) & 0x0F) - 8));
        }
    }

    // Handle remaining block
    if (kb < num_blocks) {
        const float* act_block = act_row + kb * 32;
        const uint8_t* weight_block = weight_row + kb * 18;

        half d_w;
        memcpy(&d_w, weight_block, sizeof(half));
        float dw = __half2float(d_w);
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

// Kernel for large M (shared memory, reduced sync)
__global__ void quant_gemm_q4_0_fp32_kernel_large(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_base = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_base >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    float acc = 0.0f;

    // Shared memory for activation block - load once per block iteration
    __shared__ float act_shared[Q4_0_BLOCK_SIZE];

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_base * num_blocks * 18;

    // Load activation in chunks of 4 blocks to reduce sync overhead
    int kb = 0;
    const int chunk_size = 4;

    for (; kb + chunk_size <= num_blocks; kb += chunk_size) {
        // Load 4 activation blocks into shared memory
        if (threadIdx.x < Q4_0_BLOCK_SIZE) {
            #pragma unroll
            for (int c = 0; c < chunk_size; c++) {
                act_shared[c * 32 + threadIdx.x] = act_row[(kb + c) * 32 + threadIdx.x];
            }
        }
        __syncthreads();

        // Process all 4 blocks from shared memory
        #pragma unroll
        for (int c = 0; c < chunk_size; c++) {
            const uint8_t* weight_block = weight_row + (kb + c) * 18;

            half d_w;
            memcpy(&d_w, weight_block, sizeof(half));
            float dw = __half2float(d_w);
            const uint8_t* w_qs = weight_block + 2;
            const float* as = act_shared + c * 32;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t b = w_qs[i];
                acc += as[i] * (dw * ((int)(b & 0x0F) - 8));
                acc += as[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
            }
        }

        __syncthreads();
    }

    // Handle remaining blocks
    for (; kb < num_blocks; kb++) {
        if (threadIdx.x < Q4_0_BLOCK_SIZE) {
            act_shared[threadIdx.x] = act_row[kb * 32 + threadIdx.x];
        }
        __syncthreads();

        const uint8_t* weight_block = weight_row + kb * 18;

        half d_w;
        memcpy(&d_w, weight_block, sizeof(half));
        float dw = __half2float(d_w);
        const uint8_t* w_qs = weight_block + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = w_qs[i];
            acc += act_shared[i] * (dw * ((int)(b & 0x0F) - 8));
            acc += act_shared[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
        }

        __syncthreads();
    }

    output[m_idx * N + n_base] = acc;
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

    // Strategy dispatch based on M
    if (M < 16) {
        quant_gemm_q4_0_fp32_kernel_small<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        quant_gemm_q4_0_fp32_kernel_large<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V8 Optimized)");
}
