/**
 * W4A32C8: BLOCK_Q4_0 weight x FP32 activation GEMM kernel
 * N=12288, K=5120, M varies
 *
 * v8: Combined version - best kernel per M value
 * - M=1: Warp-level with shared memory (from v6)
 * - M<=8: Vectorized loads (from v5)
 * - M>8: Simple shared memory (from v2)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK 32
#define WARP_SIZE 32

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
// M=1 Kernel: Warp-level with shared memory cache (best for M=1)
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_m1_warp(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K)
{
    extern __shared__ float s_activation[];  // Size = K

    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Cooperatively load activation into shared memory
    for (int i = tid; i < K; i += blockDim.x) {
        s_activation[i] = activation[i];
    }
    __syncthreads();

    // Each warp computes 1 output (8 warps per block)
    const int n_base = blockIdx.x * 8;
    const int n = n_base + warp;
    if (n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const block_q4_0* w_row = &w_blocks[n * num_blocks_k];

    float sum = 0.0f;

    // Process K blocks - each lane handles some blocks
    for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
        const float* act_block = &s_activation[kb * QK];
        const block_q4_0 w_block = w_row[kb];

        const float d_w = fp16_to_fp32(w_block.d);
        const uint8_t* qs = w_block.qs;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t p0 = qs[i];
            uint8_t p1 = qs[i + 8];

            int q0l = p0 & 0x0F;
            int q0h = (p0 >> 4) & 0x0F;
            int q1l = p1 & 0x0F;
            int q1h = (p1 >> 4) & 0x0F;

            sum += d_w * (q0l - 8) * act_block[i];
            sum += d_w * (q0h - 8) * act_block[i + 16];
            sum += d_w * (q1l - 8) * act_block[i + 8];
            sum += d_w * (q1h - 8) * act_block[i + 24];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[n] = sum;
    }
}

// ============================================================================
// Small batch (M<=8): Vectorized loads (best for M=2-8)
// ============================================================================
__global__ void gemm_q4_0_small_batch_vectorized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks = K / QK;

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const float* act_row = activation + row * K;

    const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
    const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);

    for (int block = 0; block < num_blocks; block++) {
        const float* act_block = act_row + block * 32;
        const block_q4_0* w_block = &w_col[block];

        // Vectorized loads for activation
        float a_vals[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 a = *reinterpret_cast<const float4*>(&act_block[i * 4]);
            a_vals[i * 4 + 0] = a.x;
            a_vals[i * 4 + 1] = a.y;
            a_vals[i * 4 + 2] = a.z;
            a_vals[i * 4 + 3] = a.w;
        }

        float d_w = fp16_to_fp32(w_block->d);
        const uint8_t* qs = w_block->qs;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            int q_low = packed & 0x0F;
            int q_high = (packed >> 4) & 0x0F;

            sum += d_w * (q_low - 8) * a_vals[i];
            sum += d_w * (q_high - 8) * a_vals[i + 16];
        }
    }

    output[row * N + col] = sum;
}

// ============================================================================
// Large batch (M>8): Shared memory for activation tiles
// ============================================================================
__global__ void gemm_q4_0_large_batch_shared(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks = K / QK;

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    __shared__ float s_act[32];

    float sum = 0.0f;
    const float* act_row = activation + row * K;

    const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
    const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);

    for (int block = 0; block < num_blocks; block++) {
        // Load activation into shared memory
        if (threadIdx.x < 32) {
            s_act[threadIdx.x] = act_row[block * 32 + threadIdx.x];
        }
        __syncthreads();

        float d_w = fp16_to_fp32(w_col[block].d);
        const uint8_t* qs = w_col[block].qs;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            int q_low = packed & 0x0F;
            int q_high = (packed >> 4) & 0x0F;

            sum += d_w * (q_low - 8) * s_act[i];
            sum += d_w * (q_high - 8) * s_act[i + 16];
        }

        __syncthreads();
    }

    output[row * N + col] = sum;
}

// ============================================================================
// Host dispatch - select best kernel per M
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

    if (M == 1) {
        // M=1: Warp-level with shared memory (best: 1.80 TFLOPS)
        const int outputs_per_block = 8;
        const int num_blocks = (N + outputs_per_block - 1) / outputs_per_block;
        const int smem_size = K * sizeof(float);

        gemm_q4_0_m1_warp<<<num_blocks, 256, smem_size>>>(
            weight_ptr, activation_ptr, output_ptr, N, K
        );
    } else if (M <= 8) {
        // M=2-8: Vectorized loads (best: 2.03 TFLOPS for M=8)
        const int threads_per_block = 256;
        const int num_blocks_x = (N + threads_per_block - 1) / threads_per_block;

        dim3 block(threads_per_block);
        dim3 grid(num_blocks_x, M);

        gemm_q4_0_small_batch_vectorized<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else {
        // M>8: Shared memory tiles (best: 2.00 TFLOPS for M=512)
        const int threads_per_block = 256;
        const int num_blocks_x = (N + threads_per_block - 1) / threads_per_block;

        dim3 block(threads_per_block);
        dim3 grid(num_blocks_x, M);

        gemm_q4_0_large_batch_shared<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM - Combined Best Kernels");
}
