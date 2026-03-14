#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q8_0 block quantization format (34 bytes per block)
// Each block contains 32 int8 quantized values with a shared FP16 scale
typedef struct {
    uint16_t d;     // FP16 scale (2 bytes)
    int8_t qs[32];  // Quantized values (32 bytes)
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block must be 34 bytes");

// Device function to safely read FP16 as float32
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * Quantized GEMM kernel: C = A @ W^T
 * A: [M, K] FP32 activation
 * W: [N, K/32] Q8_0 quantized weights (each block is 34 bytes)
 * C: [M, N] FP32 output
 *
 * This kernel computes matrix multiplication with block-quantized weights.
 * Each output element is computed as the dot product of one row of A with
 * one row of W, where the weights are stored in Q8_0 block format.
 *
 * Memory-bound optimization strategy:
 * - Vectorized loads for both activation and weight data
 * - Accumulate directly in FP32 for maximum accuracy
 * - Process multiple N elements per thread for better L2 cache utilization
 */
template <int N_PER_THREAD>
__global__ void q8_0_gemm_kernel(
    const uint8_t* __restrict__ weight_q8,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Grid-stride loop for M (batch dimension)
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M) return;

    // Each thread processes N_PER_THREAD output elements
    int n_base = blockIdx.x * blockDim.x + threadIdx.x;
    n_base *= N_PER_THREAD;

    const int num_blocks = K / 32;  // 48 blocks for K=1536

    // Accumulators for N_PER_THREAD output values
    float acc[N_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    // Compute dot product across K dimension
    for (int k_block = 0; k_block < num_blocks; ++k_block) {
        // Load activation block: [M, K] -> 32 consecutive values
        const float* act_block = &activation[m * K + k_block * 32];

        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; ++i) {
            int n = n_base + i;
            if (n < N) {
                // Get Q8_0 block for weight[n, k_block]
                const block_q8_0* w_block = reinterpret_cast<const block_q8_0*>(
                    &weight_q8[n * num_blocks * 34 + k_block * 34]
                );

                // Read FP16 scale as float32
                float d_w = read_half_as_float(w_block->d);

                // Compute dot product for this block
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    sum += act_block[j] * static_cast<float>(w_block->qs[j]);
                }
                acc[i] += sum * d_w;
            }
        }
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        int n = n_base + i;
        if (n < N) {
            output[m * N + n] = acc[i];
        }
    }
}

/**
 * Host wrapper function
 *
 * Arguments:
 *   weight: Q8_0 quantized weight tensor [N, K/32] (uint8)
 *   activation: FP32 activation tensor [M, K] (float32)
 *   M: Batch dimension
 *   N: Output features (5120)
 *   K: Input features (1536)
 */
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.scalar_type() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* weight_q8 = weight.data_ptr<uint8_t>();
    const float* act = activation.data_ptr<float>();
    float* out = output.data_ptr<float>();

    // Choose kernel configuration based on M (batch size)
    // For small M, we want more threads per output to maximize memory bandwidth utilization
    const int threads_per_block_x = 16;
    const int threads_per_block_y = 16;

    dim3 block(threads_per_block_x, threads_per_block_y);
    dim3 grid((N + 15) / 16 / 1, (M + 15) / 16);  // N_PER_THREAD = 1

    // Use N_PER_THREAD=1 for small batches (memory-bound case)
    q8_0_gemm_kernel<1><<<grid, block>>>(weight_q8, act, out, M, N, K);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (W8A32C8 Q8_0)");
}
