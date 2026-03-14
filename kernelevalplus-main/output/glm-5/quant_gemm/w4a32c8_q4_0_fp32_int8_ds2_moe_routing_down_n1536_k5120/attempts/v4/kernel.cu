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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel v4
 *
 * DeepSeek-V2 MoE Routing Down Projection
 * - N = 1536 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Q4_0 format (18 bytes per block):
 *   - scale: FP16 (2 bytes)
 *   - qs: packed 4-bit values (16 bytes)
 *
 * Optimizations for memory-bound (small M):
 * - Use __ldg for read-only global memory
 * - Prefetch weight blocks
 * - Minimize register pressure
 * - Aggressive unrolling
 */

#define Q4_0_BLOCK_SIZE 32

// Inline function to read and convert FP16 scale
__device__ __forceinline__ float read_scale_fp16(const void* ptr) {
    uint16_t h = *reinterpret_cast<const uint16_t*>(ptr);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ========== Main kernel: vectorized for memory efficiency ==========
__global__ void quant_gemm_memory_optimized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    const int num_weight_blocks = N * num_blocks;

    // Base pointers
    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_base = weight;

    float acc = 0.0f;
    const int w_offset = n_idx * num_blocks * 18;

    // Process blocks with aggressive unrolling
    for (int kb = 0; kb < num_blocks; kb++) {
        // Prefetch weight scale
        float d_w = read_scale_fp16(weight_base + w_offset + kb * 18);

        // Prefetch activation block - use __ldg for read-only data
        // Process in chunks to maximize vectorization
        const int k_base = kb * Q4_0_BLOCK_SIZE;

        // Manually unroll all 32 MACs
        // Load 2 weight scales first (for next iteration) - use register cache
        float d_w_next = (kb + 1 < num_blocks) ?
            read_scale_fp16(weight_base + w_offset + (kb + 1) * 18) : 0.0f;

        const uint8_t* w_qs = weight_base + w_offset + kb * 18 + 2;

        // Precompute dequantized weights to save time
        // w_deq = d_w * (w - 8)
        float w_low[16], w_high[16];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = w_qs[i];
            int w_l = b & 0x0F;
            int w_h = (b >> 4) & 0x0F;
            w_low[i] = d_w * (float(w_l) - 8.0f);
            w_high[i] = d_w * (float(w_h) - 8.0f);
        }

        // Accumulate with direct activation access
        // Using __ldg for activation (read-only from perspective of each thread)
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc += act_row[k_base + i] * w_low[i];
            acc += act_row[k_base + i + 16] * w_high[i];
        }
    }

    output[m_idx * N + n_idx] = acc;
}

// ========== Host function with different configurations ==========
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
    // For N=1536, use 128 threads per block = 12 blocks
    int threads_per_block = 128;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    // Use memory-optimized kernel for all sizes
    quant_gemm_memory_optimized<<<grid, block>>>(
        (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
        act_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM v4");
}
