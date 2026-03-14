/**
 * DS3 Quantized GEMM - V8
 *
 * Key insight: For M=1 with 512 outputs and only 224 K-blocks,
 * we're fundamentally limited by work available.
 *
 * Strategy: Use 512 blocks (one per output), 256 threads per block
 * All 256 threads collaborate on K dimension for one output
 * This gives 512 blocks -> 4 blocks per SM on 128-SM GPU
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int Q4_0_BLOCK = 18;
constexpr int NUM_K_BLOCKS = 224;

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float compute_block(
    const uint8_t* w_block,
    const float* act_ptr
) {
    float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));

    float a_vals[QK];
    float a_max = 0.0f;
    float a_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < QK; i += 4) {
        float4 v = *reinterpret_cast<const float4*>(act_ptr + i);
        a_vals[i] = v.x; a_vals[i+1] = v.y; a_vals[i+2] = v.z; a_vals[i+3] = v.w;
    }

    #pragma unroll
    for (int i = 0; i < QK; i++) {
        a_max = fmaxf(a_max, fabsf(a_vals[i]));
        a_sum += a_vals[i];
    }

    float d_a = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;
    float scale = 127.0f / fmaxf(a_max, 1e-10f);

    const uint8_t* qs = w_block + 2;
    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t b0 = qs[i*4+0], b1 = qs[i*4+1], b2 = qs[i*4+2], b3 = qs[i*4+3];

        int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
        int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) |
                   (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);

        int a_lo = (int)(uint8_t)lroundf(a_vals[i*4] * scale) |
                   ((int)(uint8_t)lroundf(a_vals[i*4+1] * scale) << 8) |
                   ((int)(uint8_t)lroundf(a_vals[i*4+2] * scale) << 16) |
                   ((int)(uint8_t)lroundf(a_vals[i*4+3] * scale) << 24);

        int a_hi = (int)(uint8_t)lroundf(a_vals[16+i*4] * scale) |
                   ((int)(uint8_t)lroundf(a_vals[16+i*4+1] * scale) << 8) |
                   ((int)(uint8_t)lroundf(a_vals[16+i*4+2] * scale) << 16) |
                   ((int)(uint8_t)lroundf(a_vals[16+i*4+3] * scale) << 24);

        sumi = dp4a(a_lo, w_lo, sumi);
        sumi = dp4a(a_hi, w_hi, sumi);
    }

    return d_w * (d_a * sumi - 8.0f * a_sum);
}

/**
 * One block per output, all threads collaborate on K
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

    // All threads process different K blocks
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const uint8_t* w_block = weight + (int64_t(n_idx) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
        sum += compute_block(w_block, act_row + kb * QK);
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Cross-warp reduction using shared memory
    __shared__ float warp_sums[8];
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Only warp 0 does the final reduction and write
    if (warp_id == 0) {
        float total = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane_id == 0) {
            output[output_idx] = total;
        }
    }
}

/**
 * One warp per output for medium batches
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
        const uint8_t* w_block = weight + (int64_t(n_idx) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
        sum += compute_block(w_block, act_row + kb * QK);
    }

    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[global_warp_id] = sum;
    }
}

/**
 * 2D kernel for large batches
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
        const uint8_t* w_block = weight + (int64_t(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
        sum += compute_block(w_block, act_row + kb * QK);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const int num_outputs = M * N;

    if (M == 1) {
        // Special case for M=1: one block per output
        // 512 blocks -> 4 blocks per SM (good occupancy)
        const int threads = 256;
        const int blocks = num_outputs;  // 512 for N=512

        gemm_one_block_per_output<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else if (M <= 32) {
        // Small batch: one warp per output
        const int threads = 256;
        const int blocks = (num_outputs + 7) / 8;

        gemm_warp_per_output<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        // Large batch: 2D grid
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
    m.def("forward", &forward, "Q4_0 GEMM V8");
}
