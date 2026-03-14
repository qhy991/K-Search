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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel v1
 *
 * DeepSeek-V2 MoE Up Projection
 * - N = 12288 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Q4_0 format (18 bytes per block):
 *   - scale: FP16 (2 bytes)
 *   - qs: packed 4-bit values (16 bytes, 32 values)
 * Q4_0 encoding: q = round(val / scale + 8), q ∈ [0, 15]
 * Q4_0 decoding: val = scale × (q - 8)
 *
 * Strategy: Based on Roofline analysis
 * - Small M (M < 8): Direct memory access (memory-bound, minimize overhead)
 * - Large M (M >= 8): Shared memory tiling (compute-bound, maximize throughput)
 */

#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BYTES_PER_BLOCK 18

// Helper function to read FP16 scale from bytes
__device__ __inline__ float read_fp16_scale(const uint8_t* bytes) {
    half h;
    memcpy(&h, bytes, sizeof(half));
    return __half2float(h);
}

// ========== Small M kernel: direct access, optimized for memory bandwidth ==========
__global__ void quant_gemm_small_m(
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
    const uint8_t* weight_row = weight + n_idx * num_blocks * Q4_0_BYTES_PER_BLOCK;

    // Process all blocks in K dimension
    for (int kb = 0; kb < num_blocks; kb++) {
        const float* act_block = act_row + kb * Q4_0_BLOCK_SIZE;
        const uint8_t* weight_block = weight_row + kb * Q4_0_BYTES_PER_BLOCK;

        // Read FP16 scale
        const float d_w = read_fp16_scale(weight_block);
        const uint8_t* w_qs = weight_block + 2;

        // Unroll inner loop for better instruction-level parallelism
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;

            // Dequantize and multiply-accumulate
            // w_dequant = d_w * (w - 8)
            const float w_low_f = d_w * (float(w_low) - 8.0f);
            const float w_high_f = d_w * (float(w_high) - 8.0f);

            acc += act_block[i] * w_low_f;
            acc += act_block[i + 16] * w_high_f;
        }
    }

    output[m_idx * N + n_idx] = acc;
}

// ========== Large M kernel: shared memory tiling ==========
__global__ void quant_gemm_large_m(
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

    // Shared memory for activation block (32 values)
    __shared__ float act_shared[Q4_0_BLOCK_SIZE];

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_base * num_blocks * Q4_0_BYTES_PER_BLOCK;

    // Iterate over K dimension in blocks of 32
    for (int kb = 0; kb < num_blocks; kb++) {
        // Load activation block into shared memory
        // First 32 threads load 32 values
        if (threadIdx.x < Q4_0_BLOCK_SIZE) {
            act_shared[threadIdx.x] = act_row[kb * Q4_0_BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();

        // Load weight block
        const uint8_t* weight_block = weight_row + kb * Q4_0_BYTES_PER_BLOCK;

        const float d_w = read_fp16_scale(weight_block);
        const uint8_t* w_qs = weight_block + 2;

        // Unroll for better instruction-level parallelism
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;

            const float w_low_f = d_w * (float(w_low) - 8.0f);
            const float w_high_f = d_w * (float(w_high) - 8.0f);

            // Multiply-accumulate with shared memory
            acc += act_shared[i] * w_low_f;
            acc += act_shared[i + 16] * w_high_f;
        }

        __syncthreads();
    }

    output[m_idx * N + n_base] = acc;
}

// ========== Host function with strategy dispatch ==========
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    auto weight_contig = weight.contiguous();
    auto act_contig = activation.contiguous();

    // Configure thread block
    // 256 threads per block is optimal for RTX 4090
    int threads_per_block = 256;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    // Strategy dispatch based on M (Roofline analysis)
    // Small M (M < 8): Direct access - memory-bound, minimize synchronization overhead
    // Large M (M >= 8): Shared memory - compute-bound, maximize throughput
    if (M < 8) {
        quant_gemm_small_m<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        quant_gemm_large_m<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM v1");
}
