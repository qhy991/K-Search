/**
 * Quantized GEMM for LLaMA-3-8B Attention Output with Q8_0 Weights - Final
 *
 * Optimizations:
 * 1. Vectorized float4 loads for activation (critical for memory bandwidth)
 * 2. Warp-cooperative with strided K access for small M
 * 3. Direct FP32 x INT8 computation
 * 4. Coalesced memory access pattern
 *
 * Target: RTX 4090 (CC 8.9, 128 SMs, 1008 GB/s BW)
 *
 * Q8_0 Format: 34 bytes = FP16 scale (2B) + 32 x int8 values
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

// Q8_0 block structure: 34 bytes
typedef struct {
    uint16_t d;
    int8_t qs[32];
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size must be 34 bytes");

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Kernel 1: Warp-cooperative kernel for small batch sizes (M <= 8)
 *
 * Each warp computes ONE output element.
 * All 32 lanes cooperate on K dimension with strided access.
 * Uses float4 vectorized loads for better memory bandwidth.
 */
__global__ void __launch_bounds__(256) gemm_q8_0_warp_parallel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    const int num_blocks_k = K / QK;

    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int m = idx / N;
        const int n = idx % N;

        float sum = 0.0f;

        // Strided loop: each lane processes different k-blocks
        for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + ((size_t)n * num_blocks_k + kb) * sizeof(block_q8_0)
            );

            const float d_w = read_half_as_float(wb->d);
            const float* a_ptr = activation + (size_t)m * K + kb * QK;

            // Vectorized loads using float4 (CRITICAL for memory bandwidth)
            float4 a0 = *reinterpret_cast<const float4*>(a_ptr);
            float4 a1 = *reinterpret_cast<const float4*>(a_ptr + 4);
            float4 a2 = *reinterpret_cast<const float4*>(a_ptr + 8);
            float4 a3 = *reinterpret_cast<const float4*>(a_ptr + 12);
            float4 a4 = *reinterpret_cast<const float4*>(a_ptr + 16);
            float4 a5 = *reinterpret_cast<const float4*>(a_ptr + 20);
            float4 a6 = *reinterpret_cast<const float4*>(a_ptr + 24);
            float4 a7 = *reinterpret_cast<const float4*>(a_ptr + 28);

            float block_sum = 0.0f;
            block_sum += a0.x * (float)wb->qs[0] + a0.y * (float)wb->qs[1] + a0.z * (float)wb->qs[2] + a0.w * (float)wb->qs[3];
            block_sum += a1.x * (float)wb->qs[4] + a1.y * (float)wb->qs[5] + a1.z * (float)wb->qs[6] + a1.w * (float)wb->qs[7];
            block_sum += a2.x * (float)wb->qs[8] + a2.y * (float)wb->qs[9] + a2.z * (float)wb->qs[10] + a2.w * (float)wb->qs[11];
            block_sum += a3.x * (float)wb->qs[12] + a3.y * (float)wb->qs[13] + a3.z * (float)wb->qs[14] + a3.w * (float)wb->qs[15];
            block_sum += a4.x * (float)wb->qs[16] + a4.y * (float)wb->qs[17] + a4.z * (float)wb->qs[18] + a4.w * (float)wb->qs[19];
            block_sum += a5.x * (float)wb->qs[20] + a5.y * (float)wb->qs[21] + a5.z * (float)wb->qs[22] + a5.w * (float)wb->qs[23];
            block_sum += a6.x * (float)wb->qs[24] + a6.y * (float)wb->qs[25] + a6.z * (float)wb->qs[26] + a6.w * (float)wb->qs[27];
            block_sum += a7.x * (float)wb->qs[28] + a7.y * (float)wb->qs[29] + a7.z * (float)wb->qs[30] + a7.w * (float)wb->qs[31];

            sum += d_w * block_sum;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            output[m * N + n] = sum;
        }
    }
}

/**
 * Kernel 2: 2D kernel for large batch sizes
 * Uses float4 vectorized loads for better memory bandwidth
 */
__global__ void __launch_bounds__(256) gemm_q8_0_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || m >= M) return;

    const int num_blocks_k = K / QK;
    float sum = 0.0f;
    const float* act_row = activation + (size_t)m * K;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
            weight + ((size_t)n * num_blocks_k + kb) * sizeof(block_q8_0)
        );

        const float d_w = read_half_as_float(wb->d);
        const float* a_ptr = act_row + kb * QK;

        // Vectorized loads using float4
        float4 a0 = *reinterpret_cast<const float4*>(a_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(a_ptr + 4);
        float4 a2 = *reinterpret_cast<const float4*>(a_ptr + 8);
        float4 a3 = *reinterpret_cast<const float4*>(a_ptr + 12);
        float4 a4 = *reinterpret_cast<const float4*>(a_ptr + 16);
        float4 a5 = *reinterpret_cast<const float4*>(a_ptr + 20);
        float4 a6 = *reinterpret_cast<const float4*>(a_ptr + 24);
        float4 a7 = *reinterpret_cast<const float4*>(a_ptr + 28);

        float block_sum = 0.0f;
        block_sum += a0.x * (float)wb->qs[0] + a0.y * (float)wb->qs[1] + a0.z * (float)wb->qs[2] + a0.w * (float)wb->qs[3];
        block_sum += a1.x * (float)wb->qs[4] + a1.y * (float)wb->qs[5] + a1.z * (float)wb->qs[6] + a1.w * (float)wb->qs[7];
        block_sum += a2.x * (float)wb->qs[8] + a2.y * (float)wb->qs[9] + a2.z * (float)wb->qs[10] + a2.w * (float)wb->qs[11];
        block_sum += a3.x * (float)wb->qs[12] + a3.y * (float)wb->qs[13] + a3.z * (float)wb->qs[14] + a3.w * (float)wb->qs[15];
        block_sum += a4.x * (float)wb->qs[16] + a4.y * (float)wb->qs[17] + a4.z * (float)wb->qs[18] + a4.w * (float)wb->qs[19];
        block_sum += a5.x * (float)wb->qs[20] + a5.y * (float)wb->qs[21] + a5.z * (float)wb->qs[22] + a5.w * (float)wb->qs[23];
        block_sum += a6.x * (float)wb->qs[24] + a6.y * (float)wb->qs[25] + a6.z * (float)wb->qs[26] + a6.w * (float)wb->qs[27];
        block_sum += a7.x * (float)wb->qs[28] + a7.y * (float)wb->qs[29] + a7.z * (float)wb->qs[30] + a7.w * (float)wb->qs[31];

        sum += d_w * block_sum;
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 8) {
        // Small batch: warp-parallel kernel
        int threads = 256;
        int warps_per_block = threads / WARP_SIZE;
        int total_warps = M * N;
        int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
        blocks = max(blocks, 128);  // Ensure enough blocks for GPU occupancy

        gemm_q8_0_warp_parallel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        // Large batch: 2D kernel
        dim3 block(64, 4);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        gemm_q8_0_large_m<<<grid, block>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM for LLaMA-3-8B Attention Output Final");
}
