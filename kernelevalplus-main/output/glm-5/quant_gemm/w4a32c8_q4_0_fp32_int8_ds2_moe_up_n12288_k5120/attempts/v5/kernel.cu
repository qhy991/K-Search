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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel v5 (Combined)
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
 * v5 combines the best strategies from all previous versions:
 * - M < 4: 128 threads, optimized loop unrolling (best for single_token)
 * - M < 8: 256 threads, direct access (best for small_batch)
 * - M >= 8: 256 threads, shared memory (best for large_batch)
 */

#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BYTES_PER_BLOCK 18

// Helper function to read FP16 scale from bytes
__device__ __forceinline__ float read_fp16_scale(const uint8_t* bytes) {
    uint16_t h_bits;
    memcpy(&h_bits, bytes, sizeof(uint16_t));
    return __half2float(__ushort_as_half(h_bits));
}

// ========== Very small M kernel: 128 threads, best for M=1 ==========
__global__ void quant_gemm_m_lt_4(
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

    // Process all blocks in K dimension with optimal unrolling
    for (int kb = 0; kb < num_blocks; kb++) {
        const float* act_block = act_row + kb * Q4_0_BLOCK_SIZE;
        const uint8_t* weight_block = weight_row + kb * Q4_0_BYTES_PER_BLOCK;

        const float d_w = read_fp16_scale(weight_block);
        const uint8_t* w_qs = weight_block + 2;

        // Process 32 values with 8 unrolled iterations
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;
            acc += act_block[i] * d_w * (float(w_low) - 8.0f);
            acc += act_block[i + 16] * d_w * (float(w_high) - 8.0f);
        }

        #pragma unroll
        for (int i = 8; i < 16; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;
            acc += act_block[i] * d_w * (float(w_low) - 8.0f);
            acc += act_block[i + 16] * d_w * (float(w_high) - 8.0f);
        }
    }

    output[m_idx * N + n_idx] = acc;
}

// ========== Medium-small M kernel: 256 threads, direct access ==========
__global__ void quant_gemm_m_lt_8(
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

        const float d_w = read_fp16_scale(weight_block);
        const uint8_t* w_qs = weight_block + 2;

        // Process 32 values with unrolling
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;
            acc += act_block[i] * d_w * (float(w_low) - 8.0f);
            acc += act_block[i + 16] * d_w * (float(w_high) - 8.0f);
        }
    }

    output[m_idx * N + n_idx] = acc;
}

// ========== Large M kernel: 256 threads, shared memory ==========
__global__ void quant_gemm_m_ge_8(
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
        if (threadIdx.x < Q4_0_BLOCK_SIZE) {
            act_shared[threadIdx.x] = act_row[kb * Q4_0_BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();

        // Load weight block
        const uint8_t* weight_block = weight_row + kb * Q4_0_BYTES_PER_BLOCK;

        const float d_w = read_fp16_scale(weight_block);
        const uint8_t* w_qs = weight_block + 2;

        // Process with shared memory
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;
            acc += act_shared[i] * d_w * (float(w_low) - 8.0f);
            acc += act_shared[i + 16] * d_w * (float(w_high) - 8.0f);
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

    dim3 grid, block;

    // Strategy dispatch based on M - using best kernels from all versions
    if (M < 4) {
        // Use 128 threads for M < 4 (best for single_token from v4)
        int threads_per_block = 128;
        int blocks_x = (N + threads_per_block - 1) / threads_per_block;
        int blocks_y = M;
        grid = dim3(blocks_x, blocks_y);
        block = dim3(threads_per_block);

        quant_gemm_m_lt_4<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M < 8) {
        // Use 256 threads, direct access for M < 8 (best from v1/v3)
        int threads_per_block = 256;
        int blocks_x = (N + threads_per_block - 1) / threads_per_block;
        int blocks_y = M;
        grid = dim3(blocks_x, blocks_y);
        block = dim3(threads_per_block);

        quant_gemm_m_lt_8<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Use 256 threads, shared memory for M >= 8 (best from v1/v3)
        int threads_per_block = 256;
        int blocks_x = (N + threads_per_block - 1) / threads_per_block;
        int blocks_y = M;
        grid = dim3(blocks_x, blocks_y);
        block = dim3(threads_per_block);

        quant_gemm_m_ge_8<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM v5 (Combined)");
}
