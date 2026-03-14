/**
 * W4A32C8: BLOCK_Q4_0 weight x FP32 activation GEMM kernel
 * N=12288, K=5120, M varies
 *
 * Optimizations v6:
 * 1. Activation caching in shared memory for M=1
 * 2. Warp-level processing for better parallelism
 * 3. Vectorized memory loads (float4)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK 32
#define WARP_SIZE 32

// BLOCK_Q4_0: 2 bytes FP16 scale + 16 bytes packed 4-bit values = 18 bytes
struct block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // packed 4-bit values (32 values total)
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// ============================================================================
// M=1 Kernel: Cache activation in shared memory, warp-level processing
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_m1_v6(
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

    // Phase 1: Cooperatively load activation into shared memory
    for (int i = tid; i < K; i += blockDim.x) {
        s_activation[i] = activation[i];
    }
    __syncthreads();

    // Each warp computes 1 output (8 warps per block = 8 outputs per block)
    const int n_base = blockIdx.x * 8;
    const int n = n_base + warp;
    if (n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const block_q4_0* w_row = &w_blocks[n * num_blocks_k];

    float sum = 0.0f;

    // Process K blocks - each lane in warp handles one block
    for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
        const float* act_block = &s_activation[kb * QK];
        const block_q4_0 w_block = w_row[kb];

        const float d_w = fp16_to_fp32(w_block.d);
        const uint8_t* qs = w_block.qs;

        // Unpack and compute dot product
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t p0 = qs[i];
            uint8_t p1 = qs[i + 8];

            int q_low0 = p0 & 0x0F;
            int q_high0 = (p0 >> 4) & 0x0F;
            int q_low1 = p1 & 0x0F;
            int q_high1 = (p1 >> 4) & 0x0F;

            sum += d_w * (q_low0 - 8) * act_block[i];
            sum += d_w * (q_high0 - 8) * act_block[i + 16];
            sum += d_w * (q_low1 - 8) * act_block[i + 8];
            sum += d_w * (q_high1 - 8) * act_block[i + 24];
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
// Small batch kernel (M=2-8)
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_small_batch_v6(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Grid-stride loop over (M, N) outputs
    const int total_outputs = M * N;
    for (int idx = blockIdx.x * num_warps + warp; idx < total_outputs; idx += gridDim.x * num_warps) {
        const int m = idx / N;
        const int n = idx % N;

        const float* act_row = &activation[m * K];
        const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
        const block_q4_0* w_row = &w_blocks[n * num_blocks_k];

        float sum = 0.0f;

        for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
            const float* act_block = &act_row[kb * QK];
            const block_q4_0 w_block = w_row[kb];

            const float d_w = fp16_to_fp32(w_block.d);
            const uint8_t* qs = w_block.qs;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t p0 = qs[i];
                uint8_t p1 = qs[i + 8];

                int q_low0 = p0 & 0x0F;
                int q_high0 = (p0 >> 4) & 0x0F;
                int q_low1 = p1 & 0x0F;
                int q_high1 = (p1 >> 4) & 0x0F;

                sum += d_w * (q_low0 - 8) * act_block[i];
                sum += d_w * (q_high0 - 8) * act_block[i + 16];
                sum += d_w * (q_low1 - 8) * act_block[i + 8];
                sum += d_w * (q_high1 - 8) * act_block[i + 24];
            }
        }

        // Warp reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            output[m * N + n] = sum;
        }
    }
}

// ============================================================================
// Large batch kernel (M>8): Simple thread-per-output with shared memory
// ============================================================================
__global__ void gemm_q4_0_large_batch_v6(
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

    __shared__ float s_act[QK];

    const float* act_row = &activation[m * K];
    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const block_q4_0* w_row = &w_blocks[n * num_blocks_k];

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load activation block to shared memory
        if (tid < QK) {
            s_act[tid] = act_row[kb * QK + tid];
        }
        __syncthreads();

        const float d_w = fp16_to_fp32(w_row[kb].d);
        const uint8_t* qs = w_row[kb].qs;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t p = qs[i];
            int q_low = p & 0x0F;
            int q_high = (p >> 4) & 0x0F;

            sum += d_w * (q_low - 8) * s_act[i];
            sum += d_w * (q_high - 8) * s_act[i + 16];
        }

        __syncthreads();
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

    const int num_blocks_k = K / QK;

    if (M == 1) {
        // M=1: Use shared memory for activation caching
        const int outputs_per_block = 8;
        const int num_blocks = (N + outputs_per_block - 1) / outputs_per_block;
        const int smem_size = K * sizeof(float);

        gemm_q4_0_m1_v6<<<num_blocks, 256, smem_size>>>(
            weight_ptr, activation_ptr, output_ptr, N, K
        );
    } else if (M <= 8) {
        // Small batch: Grid-stride loop
        const int num_warps_per_block = 8;
        const int threads_per_block = num_warps_per_block * WARP_SIZE;
        const int num_blocks = min(128, (M * N + num_warps_per_block - 1) / num_warps_per_block);

        gemm_q4_0_small_batch_v6<<<num_blocks, threads_per_block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else {
        // Large batch: Simple thread-per-output with shared memory
        const int threads_per_block = 256;
        const int num_blocks_x = (N + threads_per_block - 1) / threads_per_block;

        dim3 block(threads_per_block);
        dim3 grid(num_blocks_x, M);

        gemm_q4_0_large_batch_v6<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
