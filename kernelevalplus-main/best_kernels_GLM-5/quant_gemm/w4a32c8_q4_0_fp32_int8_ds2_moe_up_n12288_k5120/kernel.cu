/**
 * Optimized Quantized GEMM for DeepSeek-V2 MoE Up Projection
 * N = 12288, K = 5120, M = batch
 *
 * Strategy: Focus on memory bandwidth optimization for M=1 case
 * The key insight: Weight is 35 MB, must be read efficiently
 *
 * Optimizations:
 * 1. Use shared memory for activation caching within a block
 * 2. Coalesced memory access for weights
 * 3. Optimal thread block configuration
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int Q4_0_BLOCK = 18;
constexpr int K_BLOCKS = 160;

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ============================================================================
// Simple kernel with optimal thread configuration
// 128 threads per block, one output per thread
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q4_0_simple_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    float sum = 0.0f;

    for (int kb = 0; kb < K_BLOCKS; kb++) {
        const uint8_t* w_ptr = weight + (n * K_BLOCKS + kb) * Q4_0_BLOCK;
        const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_ptr));
        const uint8_t* qs = w_ptr + 2;
        const int k_base = kb * QK;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            int w0 = (packed & 0x0F) - 8;
            int w1 = ((packed >> 4) & 0x0F) - 8;

            sum += activation[m * K + k_base + i] * (d_w * w0);
            sum += activation[m * K + k_base + i + 16] * (d_w * w1);
        }
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Block-level shared memory kernel
// All threads in a block share activation loading
// ============================================================================
constexpr int BLOCK_THREADS = 64;
constexpr int OUTPUTS_PER_BLOCK = BLOCK_THREADS;

__global__ void __launch_bounds__(BLOCK_THREADS) gemm_q4_0_smem_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int n_base = blockIdx.x * OUTPUTS_PER_BLOCK;
    const int n = n_base + tid;

    if (m >= M || n >= N) return;

    // Shared memory for activation block (32 floats = 128 bytes)
    __shared__ float sh_act[QK];

    float sum = 0.0f;

    for (int kb = 0; kb < K_BLOCKS; kb++) {
        const int k_base = kb * QK;

        // Cooperatively load activation into shared memory
        if (tid < 8) {
            const float4 v = reinterpret_cast<const float4*>(activation + m * K + k_base)[tid];
            sh_act[tid * 4 + 0] = v.x;
            sh_act[tid * 4 + 1] = v.y;
            sh_act[tid * 4 + 2] = v.z;
            sh_act[tid * 4 + 3] = v.w;
        }
        __syncthreads();

        // Compute dot product
        const uint8_t* w_ptr = weight + (n * K_BLOCKS + kb) * Q4_0_BLOCK;
        const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_ptr));
        const uint8_t* qs = w_ptr + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            int w0 = (packed & 0x0F) - 8;
            int w1 = ((packed >> 4) & 0x0F) - 8;

            sum += sh_act[i] * (d_w * w0);
            sum += sh_act[i + 16] * (d_w * w1);
        }

        __syncthreads();
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Multi-row processing kernel
// Each block processes multiple rows to improve weight reuse
// ============================================================================
constexpr int ROWS_PER_BLOCK = 4;
constexpr int COLS_PER_BLOCK = 32;

__global__ void __launch_bounds__(ROWS_PER_BLOCK * COLS_PER_BLOCK) gemm_q4_0_multirow_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int row_in_block = tid / COLS_PER_BLOCK;  // 0-3
    const int col_in_block = tid % COLS_PER_BLOCK;  // 0-31

    const int m = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    const int n_base = blockIdx.y * COLS_PER_BLOCK;
    const int n = n_base + col_in_block;

    if (m >= M || n >= N) {
        if (tid == 0 && m < M) {
            // Handle boundary case
        }
        return;
    }

    float sum = 0.0f;

    for (int kb = 0; kb < K_BLOCKS; kb++) {
        const int k_base = kb * QK;

        // Load weight
        const uint8_t* w_ptr = weight + (n * K_BLOCKS + kb) * Q4_0_BLOCK;
        const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_ptr));
        const uint8_t* qs = w_ptr + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            int w0 = (packed & 0x0F) - 8;
            int w1 = ((packed >> 4) & 0x0F) - 8;

            sum += activation[m * K + k_base + i] * (d_w * w0);
            sum += activation[m * K + k_base + i + 16] * (d_w * w1);
        }
    }

    output[m * N + n] = sum;
}

// ============================================================================
// PyTorch Interface
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        // Single row: simple kernel with 128 threads
        dim3 block(128);
        dim3 grid((N + 127) / 128, M);

        gemm_q4_0_simple_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 32) {
        // Small batch: shared memory kernel
        dim3 block(BLOCK_THREADS);
        dim3 grid((N + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK, M);

        gemm_q4_0_smem_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: multi-row kernel for better weight reuse
        dim3 block(ROWS_PER_BLOCK * COLS_PER_BLOCK);
        dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);

        gemm_q4_0_multirow_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM DeepSeek-V2 MoE Up v8");
}
