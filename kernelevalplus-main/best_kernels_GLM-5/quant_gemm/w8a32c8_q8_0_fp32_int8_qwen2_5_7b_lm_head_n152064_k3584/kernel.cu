/**
 * Quantized GEMM for Qwen2.5-7B LM Head with Q8_0 Weights - Final
 *
 * Parameters:
 *   - N = 152064 (vocabulary size)
 *   - K = 3584 (hidden size)
 *   - M = batch size
 *
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 * Formula: result = d_w * sum(a * qs)
 *
 * Optimizations:
 *   1. Warp-cooperative kernel for small batches (M <= 8)
 *      - Each warp computes one output element
 *      - Strided K access for parallelism
 *      - Vectorized float4 loads for activation
 *   2. 2D naive kernel for large batches (M > 8)
 *      - Each thread computes one output element
 *      - Coalesced memory access pattern
 *
 * Target: RTX 4090 (CC 8.9, 128 SMs, 1008 GB/s BW)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

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
 * Kernel for M=1: High-occupancy warp-parallel
 * Each warp computes one output element.
 * Uses 512 threads per block for better SM utilization.
 */
__global__ void __launch_bounds__(512) gemm_m1_high_occupancy(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = (gridDim.x * blockDim.x) >> 5;
    const int num_blocks_k = K / QK;

    for (int n = warp_id; n < N; n += num_warps) {
        float sum = 0.0f;

        for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + ((size_t)n * num_blocks_k + b) * sizeof(block_q8_0)
            );

            const float d_w = read_half_as_float(wb->d);
            const int k_start = b * QK;
            const float* a_ptr = activation + k_start;

            const float4 a0 = *reinterpret_cast<const float4*>(a_ptr);
            const float4 a1 = *reinterpret_cast<const float4*>(a_ptr + 4);
            const float4 a2 = *reinterpret_cast<const float4*>(a_ptr + 8);
            const float4 a3 = *reinterpret_cast<const float4*>(a_ptr + 12);
            const float4 a4 = *reinterpret_cast<const float4*>(a_ptr + 16);
            const float4 a5 = *reinterpret_cast<const float4*>(a_ptr + 20);
            const float4 a6 = *reinterpret_cast<const float4*>(a_ptr + 24);
            const float4 a7 = *reinterpret_cast<const float4*>(a_ptr + 28);

            float block_sum = 0.0f;
            block_sum += a0.x * wb->qs[0] + a0.y * wb->qs[1] + a0.z * wb->qs[2] + a0.w * wb->qs[3];
            block_sum += a1.x * wb->qs[4] + a1.y * wb->qs[5] + a1.z * wb->qs[6] + a1.w * wb->qs[7];
            block_sum += a2.x * wb->qs[8] + a2.y * wb->qs[9] + a2.z * wb->qs[10] + a2.w * wb->qs[11];
            block_sum += a3.x * wb->qs[12] + a3.y * wb->qs[13] + a3.z * wb->qs[14] + a3.w * wb->qs[15];
            block_sum += a4.x * wb->qs[16] + a4.y * wb->qs[17] + a4.z * wb->qs[18] + a4.w * wb->qs[19];
            block_sum += a5.x * wb->qs[20] + a5.y * wb->qs[21] + a5.z * wb->qs[22] + a5.w * wb->qs[23];
            block_sum += a6.x * wb->qs[24] + a6.y * wb->qs[25] + a6.z * wb->qs[26] + a6.w * wb->qs[27];
            block_sum += a7.x * wb->qs[28] + a7.y * wb->qs[29] + a7.z * wb->qs[30] + a7.w * wb->qs[31];

            sum += d_w * block_sum;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            output[n] = sum;
        }
    }
}

/**
 * Kernel for small batches (2 <= M <= 8)
 */
__global__ void __launch_bounds__(256) gemm_warp_parallel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = (gridDim.x * blockDim.x) >> 5;
    const int num_blocks_k = K / QK;

    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int m = idx / N;
        const int n = idx % N;

        float sum = 0.0f;

        for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + ((size_t)n * num_blocks_k + b) * sizeof(block_q8_0)
            );

            const float d_w = read_half_as_float(wb->d);
            const int k_start = b * QK;
            const float* a_ptr = activation + (size_t)m * K + k_start;

            const float4 a0 = *reinterpret_cast<const float4*>(a_ptr);
            const float4 a1 = *reinterpret_cast<const float4*>(a_ptr + 4);
            const float4 a2 = *reinterpret_cast<const float4*>(a_ptr + 8);
            const float4 a3 = *reinterpret_cast<const float4*>(a_ptr + 12);
            const float4 a4 = *reinterpret_cast<const float4*>(a_ptr + 16);
            const float4 a5 = *reinterpret_cast<const float4*>(a_ptr + 20);
            const float4 a6 = *reinterpret_cast<const float4*>(a_ptr + 24);
            const float4 a7 = *reinterpret_cast<const float4*>(a_ptr + 28);

            float block_sum = 0.0f;
            block_sum += a0.x * wb->qs[0] + a0.y * wb->qs[1] + a0.z * wb->qs[2] + a0.w * wb->qs[3];
            block_sum += a1.x * wb->qs[4] + a1.y * wb->qs[5] + a1.z * wb->qs[6] + a1.w * wb->qs[7];
            block_sum += a2.x * wb->qs[8] + a2.y * wb->qs[9] + a2.z * wb->qs[10] + a2.w * wb->qs[11];
            block_sum += a3.x * wb->qs[12] + a3.y * wb->qs[13] + a3.z * wb->qs[14] + a3.w * wb->qs[15];
            block_sum += a4.x * wb->qs[16] + a4.y * wb->qs[17] + a4.z * wb->qs[18] + a4.w * wb->qs[19];
            block_sum += a5.x * wb->qs[20] + a5.y * wb->qs[21] + a5.z * wb->qs[22] + a5.w * wb->qs[23];
            block_sum += a6.x * wb->qs[24] + a6.y * wb->qs[25] + a6.z * wb->qs[26] + a6.w * wb->qs[27];
            block_sum += a7.x * wb->qs[28] + a7.y * wb->qs[29] + a7.z * wb->qs[30] + a7.w * wb->qs[31];

            sum += d_w * block_sum;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            output[m * N + n] = sum;
        }
    }
}

/**
 * Kernel for large batches (M > 8)
 */
__global__ void __launch_bounds__(256) gemm_2d_naive(
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

    for (int b = 0; b < num_blocks_k; b++) {
        const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
            weight + ((size_t)n * num_blocks_k + b) * sizeof(block_q8_0)
        );

        const float d_w = read_half_as_float(wb->d);
        const int k_start = b * QK;
        const float* a_ptr = activation + (size_t)m * K + k_start;

        const float4 a0 = *reinterpret_cast<const float4*>(a_ptr);
        const float4 a1 = *reinterpret_cast<const float4*>(a_ptr + 4);
        const float4 a2 = *reinterpret_cast<const float4*>(a_ptr + 8);
        const float4 a3 = *reinterpret_cast<const float4*>(a_ptr + 12);
        const float4 a4 = *reinterpret_cast<const float4*>(a_ptr + 16);
        const float4 a5 = *reinterpret_cast<const float4*>(a_ptr + 20);
        const float4 a6 = *reinterpret_cast<const float4*>(a_ptr + 24);
        const float4 a7 = *reinterpret_cast<const float4*>(a_ptr + 28);

        float block_sum = 0.0f;
        block_sum += a0.x * wb->qs[0] + a0.y * wb->qs[1] + a0.z * wb->qs[2] + a0.w * wb->qs[3];
        block_sum += a1.x * wb->qs[4] + a1.y * wb->qs[5] + a1.z * wb->qs[6] + a1.w * wb->qs[7];
        block_sum += a2.x * wb->qs[8] + a2.y * wb->qs[9] + a2.z * wb->qs[10] + a2.w * wb->qs[11];
        block_sum += a3.x * wb->qs[12] + a3.y * wb->qs[13] + a3.z * wb->qs[14] + a3.w * wb->qs[15];
        block_sum += a4.x * wb->qs[16] + a4.y * wb->qs[17] + a4.z * wb->qs[18] + a4.w * wb->qs[19];
        block_sum += a5.x * wb->qs[20] + a5.y * wb->qs[21] + a5.z * wb->qs[22] + a5.w * wb->qs[23];
        block_sum += a6.x * wb->qs[24] + a6.y * wb->qs[25] + a6.z * wb->qs[26] + a6.w * wb->qs[27];
        block_sum += a7.x * wb->qs[28] + a7.y * wb->qs[29] + a7.z * wb->qs[30] + a7.w * wb->qs[31];

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

    if (M == 1) {
        // M=1: High-occupancy kernel
        const int threads = 512;
        const int warps_per_block = threads / WARP_SIZE;
        const int num_blocks = (N + warps_per_block - 1) / warps_per_block;
        const int blocks = max(num_blocks, 384);

        gemm_m1_high_occupancy<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), N, K);
    } else if (M <= 8) {
        const int threads = 256;
        const int warps_per_block = threads / WARP_SIZE;
        const int total_warps = M * N;
        int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
        blocks = max(blocks, 256);

        gemm_warp_parallel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        dim3 block(64, 4);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        gemm_2d_naive<<<grid, block>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM for Qwen2.5-7B LM Head Final");
}
