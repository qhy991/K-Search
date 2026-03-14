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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v1 - Initial)
 *
 * Qwen3-4B FFN Down projection
 * - N = 2560 (output features)
 * - K = 9728 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Q4_0 format (18 bytes per block):
 * - d: half (2 bytes) - scale factor
 * - qs: uint8[16] (16 bytes) - 32 packed 4-bit values
 *
 * For W4A32C8, activation is FP32 (not Q8_1), so we dequantize weights
 * and multiply with FP32 activations directly.
 *
 * Computation: output = activation @ (d_w * (q_w - 8))^T
 *
 * This version implements a simple, correct baseline focused on:
 * 1. Memory-bound optimization for small M (vectorized loads)
 * 2. Compute-bound optimization for large M (shared memory tiling)
 */

#define Q4_0_BLOCK_SIZE 32

// Device function to convert uint16_t to float (FP16 half)
__device__ __inline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Small batch kernel: no shared memory, optimized for M < 16
// Direct global memory reads with minimal overhead
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

    // Process blocks with loop unrolling for better ILP
    int kb = 0;
    for (; kb + 1 < num_blocks; kb += 2) {
        const float* ab0 = act_row + kb * 32;
        const float* ab1 = ab0 + 32;

        const uint8_t* wb0 = weight_row + kb * 18;
        const uint8_t* wb1 = wb0 + 18;

        // Read scales (FP16 -> FP32)
        float dw0 = half_to_float(*(const uint16_t*)wb0);
        float dw1 = half_to_float(*(const uint16_t*)wb1);

        const uint8_t* wqs0 = wb0 + 2;
        const uint8_t* wqs1 = wb1 + 2;

        // Unroll for better throughput
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b0 = wqs0[i];
            uint8_t b1 = wqs1[i];
            // Q4_0 uses offset-8 encoding
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

        float dw = half_to_float(*(const uint16_t*)weight_block);
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

// Large batch kernel: shared memory tiling for compute-bound workloads
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

    // Load activation in chunks to reduce sync overhead
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

            float dw = half_to_float(*(const uint16_t*)weight_block);
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

        float dw = half_to_float(*(const uint16_t*)weight_block);
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

    // Strategy dispatch based on batch size
    // Small batches: memory-bound, use direct reads
    // Large batches: compute-bound, use shared memory tiling
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
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V1 - Initial)");
}
