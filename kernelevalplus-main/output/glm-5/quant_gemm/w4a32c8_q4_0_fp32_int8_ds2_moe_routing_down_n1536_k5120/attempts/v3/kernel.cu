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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel v3
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
 * Optimizations:
 * - __ldg for better global memory access
 * - Precompute weight dequantization
 * - Efficient inner loop unrolling
 */

#define Q4_0_BLOCK_SIZE 32

// Helper to read FP16 scale as float32
__device__ inline float read_fp16_scale(const uint8_t* ptr) {
    uint16_t h = *reinterpret_cast<const uint16_t*>(ptr);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ========== Main kernel: optimized for all batch sizes ==========
__global__ void quant_gemm_optimized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;

    // Pointers for this row
    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_idx * num_blocks * 18;

    float acc = 0.0f;

    // Process all blocks in K dimension
    for (int kb = 0; kb < num_blocks; kb++) {
        // Use __ldg for better caching of read-only data
        const float* act_block_ptr = act_row + kb * Q4_0_BLOCK_SIZE;

        // Read weight scale once per block
        float d_w = read_fp16_scale(weight_row + kb * 18);
        const uint8_t* w_qs = weight_row + kb * 18 + 2;

        // Process 32 values with unrolled loop
        // Pattern: w_low * act[i] + w_high * act[i+16]
        // where w_low, w_high are dequantized: d_w * (w - 8)

        // Precompute 16 weight pairs (dequantized)
        float w_low[16], w_high[16];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = w_qs[i];
            int w_low_int = byte_val & 0x0F;
            int w_high_int = (byte_val >> 4) & 0x0F;

            w_low[i] = d_w * (float(w_low_int) - 8.0f);
            w_high[i] = d_w * (float(w_high_int) - 8.0f);
        }

        // Accumulate 32 MACs
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc += act_block_ptr[i] * w_low[i];
            acc += act_block_ptr[i + 16] * w_high[i];
        }
    }

    output[m_idx * N + n_idx] = acc;
}

// ========== Host function ==========
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
    // Use 128 threads per block for better occupancy
    int threads_per_block = 128;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    quant_gemm_optimized<<<grid, block>>>(
        (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
        act_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM v3 - Optimized");
}
