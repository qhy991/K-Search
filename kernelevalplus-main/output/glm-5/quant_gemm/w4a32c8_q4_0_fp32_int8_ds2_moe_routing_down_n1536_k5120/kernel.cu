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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel - Final Combined Version
 *
 * DeepSeek-V2 MoE Routing Down Projection
 * - N = 1536 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Q4_0 format (18 bytes per block):
 *   - scale: FP16 (2 bytes)
 *   - qs: packed 4-bit values (16 bytes)
 * Q4_0 encoding: q = round(val / scale + 8), q ∈ [0, 15]
 * Q4_0 decoding: val = scale × (q - 8)
 *
 * Strategy Dispatch:
 * - Small M (M <= 8): Memory-bound kernel - direct access, minimal overhead
 * - Large M (M > 8): Compute-bound kernel - optimized for throughput
 */

#define Q4_0_BLOCK_SIZE 32

// Helper to read FP16 scale as float32
__device__ __forceinline__ float read_scale_fp16(const void* ptr) {
    uint16_t h = *reinterpret_cast<const uint16_t*>(ptr);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ========== Small M kernel: optimized for memory bandwidth ==========
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
    const uint8_t* weight_row = weight + n_idx * num_blocks * 18;

    // Process all blocks in K dimension with aggressive unrolling
    for (int kb = 0; kb < num_blocks; kb++) {
        const int k_base = kb * Q4_0_BLOCK_SIZE;

        // Read weight scale once
        float d_w = read_scale_fp16(weight_row + kb * 18);
        const uint8_t* w_qs = weight_row + kb * 18 + 2;

        // Precompute dequantized weights for this block
        // w_deq = d_w * (w - 8)
        float w_low[16], w_high[16];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = w_qs[i];
            w_low[i] = d_w * (float(b & 0x0F) - 8.0f);
            w_high[i] = d_w * (float((b >> 4) & 0x0F) - 8.0f);
        }

        // Accumulate 32 MACs with unrolled loop
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc += act_row[k_base + i] * w_low[i];
            acc += act_row[k_base + i + 16] * w_high[i];
        }
    }

    output[m_idx * N + n_idx] = acc;
}

// ========== Large M kernel: optimized for compute throughput ==========
__global__ void quant_gemm_large_m(
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

    // Process all blocks in K dimension
    for (int kb = 0; kb < num_blocks; kb++) {
        const int k_base = kb * Q4_0_BLOCK_SIZE;

        // Read weight scale once
        float d_w = read_scale_fp16(weight_row + kb * 18);
        const uint8_t* w_qs = weight_row + kb * 18 + 2;

        // Precompute dequantized weights
        float w_low[16], w_high[16];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = w_qs[i];
            w_low[i] = d_w * (float(b & 0x0F) - 8.0f);
            w_high[i] = d_w * (float((b >> 4) & 0x0F) - 8.0f);
        }

        // Accumulate 32 MACs
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc += act_row[k_base + i] * w_low[i];
            acc += act_row[k_base + i + 16] * w_high[i];
        }
    }

    output[m_idx * N + n_idx] = acc;
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
    // 128 threads per block is optimal for RTX 4090
    int threads_per_block = 128;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    // Strategy dispatch based on M (Roofline analysis)
    // Small M (M <= 8): Memory-bound - use simpler kernel with less overhead
    // Large M (M > 8): Compute-bound - use same optimized kernel structure
    if (M <= 8) {
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
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM - Final Combined");
}
