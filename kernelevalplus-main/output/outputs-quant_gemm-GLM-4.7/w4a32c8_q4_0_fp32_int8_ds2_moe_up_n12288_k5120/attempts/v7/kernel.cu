/**
 * W4A32C8: BLOCK_Q4_0 weight x FP32 activation GEMM kernel
 * N=12288, K=5120, M varies
 *
 * v7: Optimized memory access patterns
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK 32

// BLOCK_Q4_0: 2 bytes FP16 scale + 16 bytes packed 4-bit values
struct block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // packed 4-bit values
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// ============================================================================
// M=1 Kernel: Shared memory for activation, optimized grid
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_m1_v7(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K)
{
    extern __shared__ float s_activation[];  // Size = K

    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int n = blockIdx.x * blockDim.x + tid;

    if (n >= N) return;

    // Phase 1: Cooperatively load activation into shared memory
    for (int i = tid; i < K; i += blockDim.x) {
        s_activation[i] = activation[i];
    }
    __syncthreads();

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const block_q4_0* w_row = &w_blocks[n * num_blocks_k];

    float sum = 0.0f;

    // Process K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const float* act_block = &s_activation[kb * QK];
        const block_q4_0 w_block = w_row[kb];

        const float d_w = fp16_to_fp32(w_block.d);
        const uint8_t* qs = w_block.qs;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t p = qs[i];
            int ql = p & 0x0F;
            int qh = (p >> 4) & 0x0F;

            sum += d_w * (ql - 8) * act_block[i];
            sum += d_w * (qh - 8) * act_block[i + 16];
        }
    }

    output[n] = sum;
}

// ============================================================================
// Small batch (M=2-8): Simple thread-per-output
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_small_batch_v7(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + tid;

    if (m >= M || n >= N) return;

    const float* act_row = &activation[m * K];
    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const block_q4_0* w_row = &w_blocks[n * num_blocks_k];

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const float* act_block = &act_row[kb * QK];
        const block_q4_0 w_block = w_row[kb];

        const float d_w = fp16_to_fp32(w_block.d);
        const uint8_t* qs = w_block.qs;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t p = qs[i];
            int ql = p & 0x0F;
            int qh = (p >> 4) & 0x0F;

            sum += d_w * (ql - 8) * act_block[i];
            sum += d_w * (qh - 8) * act_block[i + 16];
        }
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Large batch (M>8): Simple thread-per-output
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_large_batch_v7(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + tid;

    if (m >= M || n >= N) return;

    const float* act_row = &activation[m * K];
    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const block_q4_0* w_row = &w_blocks[n * num_blocks_k];

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const float* act_block = &act_row[kb * QK];
        const block_q4_0 w_block = w_row[kb];

        const float d_w = fp16_to_fp32(w_block.d);
        const uint8_t* qs = w_block.qs;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t p = qs[i];
            int ql = p & 0x0F;
            int qh = (p >> 4) & 0x0F;

            sum += d_w * (ql - 8) * act_block[i];
            sum += d_w * (qh - 8) * act_block[i + 16];
        }
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Host dispatch
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K)
{
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* activation_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    const int threads_per_block = 256;

    if (M == 1) {
        // M=1: Shared memory for activation
        const int num_blocks = (N + threads_per_block - 1) / threads_per_block;
        const int smem_size = K * sizeof(float);

        gemm_q4_0_m1_v7<<<num_blocks, threads_per_block, smem_size>>>(
            weight_ptr, activation_ptr, output_ptr, N, K
        );
    } else if (M <= 8) {
        // Small batch
        const int num_blocks_x = (N + threads_per_block - 1) / threads_per_block;

        dim3 block(threads_per_block);
        dim3 grid(num_blocks_x, M);

        gemm_q4_0_small_batch_v7<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else {
        // Large batch
        const int num_blocks_x = (N + threads_per_block - 1) / threads_per_block;

        dim3 block(threads_per_block);
        dim3 grid(num_blocks_x, M);

        gemm_q4_0_large_batch_v7<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
