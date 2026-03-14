/**
 * Quantized GEMM for DeepSeek-V3 MoE Routing Up Projection with Q4_1 Weights
 * Target: N=7168, K=2048, NUM_K_BLOCKS=64
 *
 * Final optimized version v15 (best from all iterations):
 * - __ldg for L2 cache hints on weights
 * - Fully unrolled computation
 * - Warp-per-output for M<=32 (best M=1: 1819 GFLOPS)
 * - 2D grid for M>32
 *
 * Performance summary:
 * - M=1: 1819 GFLOPS (43% of baseline)
 * - M=512: 2610 GFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cstdint>

constexpr int QK4_1 = 32;
constexpr int WARP_SIZE = 32;
constexpr int NUM_K_BLOCKS = 64;

struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size must be 20 bytes");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Kernel 1: Warp per output for M<=32
// ============================================================================
__global__ void __launch_bounds__(128) gemm_warp_per_output(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;

    if (global_warp_id >= M * N) return;

    const int row = global_warp_id / N;
    const int col = global_warp_id % N;

    float sum = 0.0f;
    const float* act_row = activation + (int64_t)row * K;
    const block_q4_1* w_row = weight + (int64_t)col * NUM_K_BLOCKS;

    #pragma unroll 2
    for (int b = lane_id; b < NUM_K_BLOCKS; b += WARP_SIZE) {
        const block_q4_1* w_block = &w_row[b];

        const float d_w = half_to_float(__ldg(&w_block->d));
        const float m_w = half_to_float(__ldg(&w_block->m));
        const int k_start = b * QK4_1;

        const float4 a4_0 = *reinterpret_cast<const float4*>(&act_row[k_start]);
        const float4 a4_1 = *reinterpret_cast<const float4*>(&act_row[k_start + 4]);
        const float4 a4_2 = *reinterpret_cast<const float4*>(&act_row[k_start + 8]);
        const float4 a4_3 = *reinterpret_cast<const float4*>(&act_row[k_start + 12]);
        const float4 a4_4 = *reinterpret_cast<const float4*>(&act_row[k_start + 16]);
        const float4 a4_5 = *reinterpret_cast<const float4*>(&act_row[k_start + 20]);
        const float4 a4_6 = *reinterpret_cast<const float4*>(&act_row[k_start + 24]);
        const float4 a4_7 = *reinterpret_cast<const float4*>(&act_row[k_start + 28]);

        const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[0]));
        const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[4]));
        const uint32_t w2 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[8]));
        const uint32_t w3 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[12]));

        sum += (d_w * (w0 & 0xF) + m_w) * a4_0.x;
        sum += (d_w * ((w0 >> 4) & 0xF) + m_w) * a4_4.x;
        sum += (d_w * ((w0 >> 8) & 0xF) + m_w) * a4_0.y;
        sum += (d_w * ((w0 >> 12) & 0xF) + m_w) * a4_4.y;
        sum += (d_w * ((w0 >> 16) & 0xF) + m_w) * a4_0.z;
        sum += (d_w * ((w0 >> 20) & 0xF) + m_w) * a4_4.z;
        sum += (d_w * ((w0 >> 24) & 0xF) + m_w) * a4_0.w;
        sum += (d_w * ((w0 >> 28) & 0xF) + m_w) * a4_4.w;

        sum += (d_w * (w1 & 0xF) + m_w) * a4_1.x;
        sum += (d_w * ((w1 >> 4) & 0xF) + m_w) * a4_5.x;
        sum += (d_w * ((w1 >> 8) & 0xF) + m_w) * a4_1.y;
        sum += (d_w * ((w1 >> 12) & 0xF) + m_w) * a4_5.y;
        sum += (d_w * ((w1 >> 16) & 0xF) + m_w) * a4_1.z;
        sum += (d_w * ((w1 >> 20) & 0xF) + m_w) * a4_5.z;
        sum += (d_w * ((w1 >> 24) & 0xF) + m_w) * a4_1.w;
        sum += (d_w * ((w1 >> 28) & 0xF) + m_w) * a4_5.w;

        sum += (d_w * (w2 & 0xF) + m_w) * a4_2.x;
        sum += (d_w * ((w2 >> 4) & 0xF) + m_w) * a4_6.x;
        sum += (d_w * ((w2 >> 8) & 0xF) + m_w) * a4_2.y;
        sum += (d_w * ((w2 >> 12) & 0xF) + m_w) * a4_6.y;
        sum += (d_w * ((w2 >> 16) & 0xF) + m_w) * a4_2.z;
        sum += (d_w * ((w2 >> 20) & 0xF) + m_w) * a4_6.z;
        sum += (d_w * ((w2 >> 24) & 0xF) + m_w) * a4_2.w;
        sum += (d_w * ((w2 >> 28) & 0xF) + m_w) * a4_6.w;

        sum += (d_w * (w3 & 0xF) + m_w) * a4_3.x;
        sum += (d_w * ((w3 >> 4) & 0xF) + m_w) * a4_7.x;
        sum += (d_w * ((w3 >> 8) & 0xF) + m_w) * a4_3.y;
        sum += (d_w * ((w3 >> 12) & 0xF) + m_w) * a4_7.y;
        sum += (d_w * ((w3 >> 16) & 0xF) + m_w) * a4_3.z;
        sum += (d_w * ((w3 >> 20) & 0xF) + m_w) * a4_7.z;
        sum += (d_w * ((w3 >> 24) & 0xF) + m_w) * a4_3.w;
        sum += (d_w * ((w3 >> 28) & 0xF) + m_w) * a4_7.w;
    }

    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[(int64_t)row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 2: 2D grid for large M (compute-bound)
// ============================================================================
__global__ void __launch_bounds__(256) gemm_2d_grid(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    float sum = 0.0f;
    const float* act_row = activation + (int64_t)m * K;
    const block_q4_1* w_row = weight + (int64_t)n * NUM_K_BLOCKS;

    for (int b = 0; b < NUM_K_BLOCKS; ++b) {
        const block_q4_1* w_block = &w_row[b];

        const float d_w = half_to_float(__ldg(&w_block->d));
        const float m_w = half_to_float(__ldg(&w_block->m));
        const int k_start = b * QK4_1;

        const float4 a4_0 = *reinterpret_cast<const float4*>(&act_row[k_start]);
        const float4 a4_1 = *reinterpret_cast<const float4*>(&act_row[k_start + 4]);
        const float4 a4_2 = *reinterpret_cast<const float4*>(&act_row[k_start + 8]);
        const float4 a4_3 = *reinterpret_cast<const float4*>(&act_row[k_start + 12]);
        const float4 a4_4 = *reinterpret_cast<const float4*>(&act_row[k_start + 16]);
        const float4 a4_5 = *reinterpret_cast<const float4*>(&act_row[k_start + 20]);
        const float4 a4_6 = *reinterpret_cast<const float4*>(&act_row[k_start + 24]);
        const float4 a4_7 = *reinterpret_cast<const float4*>(&act_row[k_start + 28]);

        const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[0]));
        const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[4]));
        const uint32_t w2 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[8]));
        const uint32_t w3 = __ldg(reinterpret_cast<const uint32_t*>(&w_block->qs[12]));

        sum += (d_w * (w0 & 0xF) + m_w) * a4_0.x;
        sum += (d_w * ((w0 >> 4) & 0xF) + m_w) * a4_4.x;
        sum += (d_w * ((w0 >> 8) & 0xF) + m_w) * a4_0.y;
        sum += (d_w * ((w0 >> 12) & 0xF) + m_w) * a4_4.y;
        sum += (d_w * ((w0 >> 16) & 0xF) + m_w) * a4_0.z;
        sum += (d_w * ((w0 >> 20) & 0xF) + m_w) * a4_4.z;
        sum += (d_w * ((w0 >> 24) & 0xF) + m_w) * a4_0.w;
        sum += (d_w * ((w0 >> 28) & 0xF) + m_w) * a4_4.w;

        sum += (d_w * (w1 & 0xF) + m_w) * a4_1.x;
        sum += (d_w * ((w1 >> 4) & 0xF) + m_w) * a4_5.x;
        sum += (d_w * ((w1 >> 8) & 0xF) + m_w) * a4_1.y;
        sum += (d_w * ((w1 >> 12) & 0xF) + m_w) * a4_5.y;
        sum += (d_w * ((w1 >> 16) & 0xF) + m_w) * a4_1.z;
        sum += (d_w * ((w1 >> 20) & 0xF) + m_w) * a4_5.z;
        sum += (d_w * ((w1 >> 24) & 0xF) + m_w) * a4_1.w;
        sum += (d_w * ((w1 >> 28) & 0xF) + m_w) * a4_5.w;

        sum += (d_w * (w2 & 0xF) + m_w) * a4_2.x;
        sum += (d_w * ((w2 >> 4) & 0xF) + m_w) * a4_6.x;
        sum += (d_w * ((w2 >> 8) & 0xF) + m_w) * a4_2.y;
        sum += (d_w * ((w2 >> 12) & 0xF) + m_w) * a4_6.y;
        sum += (d_w * ((w2 >> 16) & 0xF) + m_w) * a4_2.z;
        sum += (d_w * ((w2 >> 20) & 0xF) + m_w) * a4_6.z;
        sum += (d_w * ((w2 >> 24) & 0xF) + m_w) * a4_2.w;
        sum += (d_w * ((w2 >> 28) & 0xF) + m_w) * a4_6.w;

        sum += (d_w * (w3 & 0xF) + m_w) * a4_3.x;
        sum += (d_w * ((w3 >> 4) & 0xF) + m_w) * a4_7.x;
        sum += (d_w * ((w3 >> 8) & 0xF) + m_w) * a4_3.y;
        sum += (d_w * ((w3 >> 12) & 0xF) + m_w) * a4_7.y;
        sum += (d_w * ((w3 >> 16) & 0xF) + m_w) * a4_3.z;
        sum += (d_w * ((w3 >> 20) & 0xF) + m_w) * a4_7.z;
        sum += (d_w * ((w3 >> 24) & 0xF) + m_w) * a4_3.w;
        sum += (d_w * ((w3 >> 28) & 0xF) + m_w) * a4_7.w;
    }

    output[(int64_t)m * N + n] = sum;
}

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const block_q4_1* weight_ptr = reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>());

    if (M <= 32) {
        // Small batch: warp per output
        const int total_outputs = M * N;
        const int threads = 128;
        const int warps_per_block = threads / WARP_SIZE;
        const int num_blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

        gemm_warp_per_output<<<num_blocks, threads>>>(
            weight_ptr,
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        // Large batch: 2D grid
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);

        gemm_2d_grid<<<grid, block>>>(
            weight_ptr,
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM DS3 MoE Routing Up N7168 K2048 Final");
}
