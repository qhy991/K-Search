/**
 * DS3 Quantized GEMM - Q4_1 × FP32 Implementation v6
 *
 * Optimized version with:
 * - Simpler vectorized loads
 * - Better loop unrolling
 *
 * Q4_1 format (20 bytes per block):
 *   - Bytes 0-1: scale (fp16)
 *   - Bytes 2-3: min (fp16)
 *   - Bytes 4-19: 16 bytes packed 4-bit values (UNSIGNED [0, 15])
 *
 * Parameters: N=512, K=7168, NUM_K_BLOCKS=224
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int Q4_1_BLOCK = 20;
constexpr int NUM_K_BLOCKS = 224;

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Optimized compute block
 */
__device__ __forceinline__ float compute_block_q4_1_fp32(
    const uint8_t* w_block,
    const float* act_ptr
) {
    // Load Q4_1 scale and min
    float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
    float m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));

    // Load activation values using vectorized loads
    float4 a0 = *reinterpret_cast<const float4*>(act_ptr);
    float4 a1 = *reinterpret_cast<const float4*>(act_ptr + 4);
    float4 a2 = *reinterpret_cast<const float4*>(act_ptr + 8);
    float4 a3 = *reinterpret_cast<const float4*>(act_ptr + 12);
    float4 a4 = *reinterpret_cast<const float4*>(act_ptr + 16);
    float4 a5 = *reinterpret_cast<const float4*>(act_ptr + 20);
    float4 a6 = *reinterpret_cast<const float4*>(act_ptr + 24);
    float4 a7 = *reinterpret_cast<const float4*>(act_ptr + 28);

    const uint8_t* qs = w_block + 4;

    float sum_a_wqs = 0.0f;
    float sum_a = 0.0f;

    // Unroll the loop for better ILP
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t b = qs[i];
        float w_low = (float)(b & 0xF);
        float w_high = (float)((b >> 4) & 0xF);

        // Get activation values
        float a_low, a_high;
        const float* ap = act_ptr + i;
        const float* aph = act_ptr + 16 + i;

        // Manual unrolling for first 4 iterations
        if (i < 4) {
            if (i == 0) { a_low = a0.x; a_high = a4.x; }
            else if (i == 1) { a_low = a0.y; a_high = a4.y; }
            else if (i == 2) { a_low = a0.z; a_high = a4.z; }
            else { a_low = a0.w; a_high = a4.w; }
        } else if (i < 8) {
            int j = i - 4;
            if (j == 0) { a_low = a1.x; a_high = a5.x; }
            else if (j == 1) { a_low = a1.y; a_high = a5.y; }
            else if (j == 2) { a_low = a1.z; a_high = a5.z; }
            else { a_low = a1.w; a_high = a5.w; }
        } else if (i < 12) {
            int j = i - 8;
            if (j == 0) { a_low = a2.x; a_high = a6.x; }
            else if (j == 1) { a_low = a2.y; a_high = a6.y; }
            else if (j == 2) { a_low = a2.z; a_high = a6.z; }
            else { a_low = a2.w; a_high = a6.w; }
        } else {
            int j = i - 12;
            if (j == 0) { a_low = a3.x; a_high = a7.x; }
            else if (j == 1) { a_low = a3.y; a_high = a7.y; }
            else if (j == 2) { a_low = a3.z; a_high = a7.z; }
            else { a_low = a3.w; a_high = a7.w; }
        }

        sum_a_wqs += a_low * w_low + a_high * w_high;
        sum_a += a_low + a_high;
    }

    return d_w * sum_a_wqs + m_w * sum_a;
}

/**
 * Strategy 1: One block per output (best for M=1)
 */
__global__ void __launch_bounds__(256) gemm_one_block_per_output(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int output_idx = blockIdx.x;
    if (output_idx >= M * N) return;

    const int m_idx = output_idx / N;
    const int n_idx = output_idx % N;

    float sum = 0.0f;
    const float* act_row = activation + m_idx * K;

    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const uint8_t* w_block = weight + (int64_t(n_idx) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
        sum += compute_block_q4_1_fp32(w_block, act_row + kb * QK);
    }

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[8];
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane_id == 0) {
            output[output_idx] = total;
        }
    }
}

/**
 * Strategy 2: One warp per output (best for small-medium batches)
 */
__global__ void __launch_bounds__(256) gemm_warp_per_output(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;

    if (global_warp_id >= M * N) return;

    const int m_idx = global_warp_id / N;
    const int n_idx = global_warp_id % N;

    float sum = 0.0f;
    const float* act_row = activation + m_idx * K;

    for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
        const uint8_t* w_block = weight + (int64_t(n_idx) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
        sum += compute_block_q4_1_fp32(w_block, act_row + kb * QK);
    }

    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[global_warp_id] = sum;
    }
}

/**
 * Strategy 3: 2D grid (best for large batches)
 */
__global__ void __launch_bounds__(256) gemm_2d_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    float sum = 0.0f;
    const float* act_row = activation + m * K;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const uint8_t* w_block = weight + (int64_t(n) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
        sum += compute_block_q4_1_fp32(w_block, act_row + kb * QK);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const int num_outputs = M * N;

    if (M == 1) {
        const int threads = 256;
        const int blocks = num_outputs;
        gemm_one_block_per_output<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else if (M <= 32) {
        const int threads = 256;
        const int blocks = (num_outputs + 7) / 8;
        gemm_warp_per_output<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_2d_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM DS3 v6");
}
