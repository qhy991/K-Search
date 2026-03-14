/**
 * W8A32C8: Q8_0 weight x FP32 activation GEMM kernel - v11 Combined Best
 *
 * DeepSeek-V2 Attention Output projection: N=5120, K=5120
 *
 * Q8_0 format:
 *   - 34 bytes per block (2 bytes FP16 scale + 32 INT8 values)
 *   - Dequantization: w = q * d
 *
 * Best configuration:
 * - M=1: 128 threads (4 warps per block) - best M=1 performance
 * - M>1: 256 threads with aggressive unrolling
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define QK 32
#define BLOCK_Q8_0_SIZE 34
#define WARP_SIZE 32

// ============================================================================
// Helper: Decode FP16 from bytes using union for safety
// ============================================================================
__device__ __forceinline__ float half_to_float_fast(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// ============================================================================
// Kernel: M=1 - Best performing configuration (128 threads)
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q8_0_fp32_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int n = blockIdx.x * warps_per_block + warp_id;
    if (n >= N) return;

    const float* act_row = activation;
    const uint8_t* w_row = weight + (long long)n * num_blocks_k * BLOCK_Q8_0_SIZE;

    float sum = 0.0f;

    // Process all K blocks with loop unrolling
    #pragma unroll 8
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const uint8_t* w_block = w_row + kb * BLOCK_Q8_0_SIZE;

        float d_w = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(w_block + 2);
        const float* act_block = act_row + kb * QK;

        float a = act_block[lane_id];
        int8_t w_q = qs[lane_id];
        sum += a * (float)w_q * d_w;
    }

    // Warp reduction with shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
        output[n] = sum;
    }
}

// ============================================================================
// Kernel: M>1 - Optimized for batch processing
// ============================================================================
__global__ void gemm_q8_0_fp32_mlarge_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int n = blockIdx.x * warps_per_block + warp_id;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    const float* act_row = activation + (long long)m * K;
    const uint8_t* w_row = weight + (long long)n * num_blocks_k * BLOCK_Q8_0_SIZE;

    float sum = 0.0f;

    // Process all K blocks
    #pragma unroll 8
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const uint8_t* w_block = w_row + kb * BLOCK_Q8_0_SIZE;

        float d_w = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(w_block + 2);
        const float* act_block = act_row + kb * QK;

        float a = act_block[lane_id];
        int8_t w_q = qs[lane_id];
        sum += a * (float)w_q * d_w;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
        output[(long long)m * N + n] = sum;
    }
}

// ============================================================================
// Host wrapper
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K)
{
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if (M == 1) {
        // M=1: Best config - 128 threads
        const int threads = 128;
        const int warps_per_block = threads / WARP_SIZE;
        const int grid = (N + warps_per_block - 1) / warps_per_block;

        gemm_q8_0_fp32_m1_kernel<<<grid, threads>>>(
            weight_ptr, act_ptr, output_ptr, N, K);
    } else {
        // M>1: 256 threads
        const int threads = 256;
        const int warps_per_block = threads / WARP_SIZE;
        const int grid_n = (N + warps_per_block - 1) / warps_per_block;

        dim3 grid(grid_n, M);
        dim3 block(threads);

        gemm_q8_0_fp32_mlarge_kernel<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, M, N, K);
    }

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 x FP32 GEMM");
}
