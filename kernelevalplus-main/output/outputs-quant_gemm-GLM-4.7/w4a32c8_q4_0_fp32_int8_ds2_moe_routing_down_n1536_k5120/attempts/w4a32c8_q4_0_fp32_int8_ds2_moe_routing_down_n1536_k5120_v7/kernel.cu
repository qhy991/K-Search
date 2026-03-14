/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel - Optimized V7
 *
 * DeepSeek-V2 MoE Routing Down projection: N=1536, K=5120
 *
 * Q4_0 format:
 *   - 18 bytes per block (2 bytes FP16 scale + 16 bytes packed 4-bit values)
 *   - Unpacking: llama.cpp style (all low nibbles 0-15, then high nibbles 16-31)
 *   - Dequantization: w = d_w × (q - 8), where q ∈ [0, 15]
 *
 * V7 Optimizations:
 *   - Each thread block computes 4 output elements
 *   - Better utilization of memory bandwidth
 *   - Reduced kernel launch overhead
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define QK 32
#define BLOCK_Q4_0_SIZE 18
#define WARP_SIZE 32

// ============================================================================
// Helper: Decode FP16 from bytes
// ============================================================================
__device__ __forceinline__ float half_to_float_fast(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// ============================================================================
// Process one Q4_0 block and return partial sum
// ============================================================================
__device__ __forceinline__ float process_q4_0_block(
    const uint8_t* w_block,
    const float* act_block,
    int lane_id
) {
    float d_w = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block));
    const uint8_t* packed = w_block + 2;

    int q;
    float a;
    if (lane_id < 16) {
        q = (packed[lane_id] & 0x0F);
        a = act_block[lane_id];
    } else {
        q = ((packed[lane_id - 16] >> 4) & 0x0F);
        a = act_block[lane_id];
    }

    return a * d_w * static_cast<float>(q - 8);
}

// ============================================================================
// Kernel: M=1 - Each thread block processes 4 consecutive N values
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q4_0_fp32_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K
) {
    const int num_blocks_k = K / QK;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Each thread block processes 4 consecutive N values (4 warps)
    const int n_base = blockIdx.x * 4;
    if (n_base >= N) return;

    // Determine which N this warp handles (0-3)
    const int warp_n = n_base + warp_id;
    const bool valid = warp_n < N;

    const float* act_row = activation;
    const uint8_t* w_row_base = weight + (long long)n_base * num_blocks_k * BLOCK_Q4_0_SIZE;

    float sum = 0.0f;

    if (valid) {
        const uint8_t* w_row = w_row_base + warp_id * num_blocks_k * BLOCK_Q4_0_SIZE;

        for (int kb = 0; kb < num_blocks_k; kb++) {
            sum += process_q4_0_block(
                w_row + kb * BLOCK_Q4_0_SIZE,
                act_row + kb * QK,
                lane_id
            );
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0 && valid) {
        output[warp_n] = sum;
    }
}

// ============================================================================
// Kernel: M>1 - Each thread block processes 4 consecutive N values
// ============================================================================
__global__ void gemm_q4_0_fp32_mlarge_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / QK;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    const int n_base = blockIdx.x * 4;
    const int m = blockIdx.y;

    if (m >= M || n_base >= N) return;

    const int warp_n = n_base + warp_id;
    const bool valid = warp_n < N;

    const float* act_row = activation + (long long)m * K;
    const uint8_t* w_row_base = weight + (long long)n_base * num_blocks_k * BLOCK_Q4_0_SIZE;

    float sum = 0.0f;

    if (valid) {
        const uint8_t* w_row = w_row_base + warp_id * num_blocks_k * BLOCK_Q4_0_SIZE;

        for (int kb = 0; kb < num_blocks_k; kb++) {
            sum += process_q4_0_block(
                w_row + kb * BLOCK_Q4_0_SIZE,
                act_row + kb * QK,
                lane_id
            );
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0 && valid) {
        output[(long long)m * N + warp_n] = sum;
    }
}

// ============================================================================
// Host wrapper
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if (M == 1) {
        const int threads = 128;
        const int grid = (N + 3) / 4;  // 4 N values per block

        gemm_q4_0_fp32_m1_kernel<<<grid, threads>>>(
            weight_ptr, act_ptr, output_ptr, N, K);
    } else {
        const int threads = 128;
        const int grid_n = (N + 3) / 4;

        dim3 grid(grid_n, M);
        dim3 block(threads);

        gemm_q4_0_fp32_mlarge_kernel<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, M, N, K);
    }

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
