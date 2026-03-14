/**
 * Optimized Q4_0 GEMM with focus on memory bandwidth
 *
 * Key insight: For M=1, the weight matrix (8.26 MB) can partially fit in L2 cache (6 MB)
 * Strategy: Process outputs in tiles to maximize weight data reuse in L2 cache
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int Q4_0_BLOCK = 18;
constexpr int NUM_K_BLOCKS = 64;

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const half*>(&h));
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Fast block dot product with minimal register usage
 */
__device__ __forceinline__ float dot_block_fast(
    const uint8_t* __restrict__ w_block,
    const float* __restrict__ a_ptr
) {
    float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
    const uint8_t* qs = w_block + 2;

    // Load activation with float4
    float a_max = 0.0f, a_sum = 0.0f;
    float av[32];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float4 v = __ldg(reinterpret_cast<const float4*>(a_ptr + i * 4));
        av[i*4] = v.x; av[i*4+1] = v.y; av[i*4+2] = v.z; av[i*4+3] = v.w;
        a_max = fmaxf(a_max, fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fmaxf(fabsf(v.z), fabsf(v.w)))));
        a_sum += v.x + v.y + v.z + v.w;
    }

    const float scale = 127.0f / fmaxf(a_max, 1e-10f);
    const float d_a = a_max / 127.0f;

    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t b0 = qs[i * 4 + 0];
        uint8_t b1 = qs[i * 4 + 1];
        uint8_t b2 = qs[i * 4 + 2];
        uint8_t b3 = qs[i * 4 + 3];

        int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
        int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) | (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);

        int a_lo = ((uint8_t)__float2int_rn(av[i*4] * scale)) |
                   (((uint8_t)__float2int_rn(av[i*4+1] * scale)) << 8) |
                   (((uint8_t)__float2int_rn(av[i*4+2] * scale)) << 16) |
                   (((uint8_t)__float2int_rn(av[i*4+3] * scale)) << 24);
        int a_hi = ((uint8_t)__float2int_rn(av[16+i*4] * scale)) |
                   (((uint8_t)__float2int_rn(av[16+i*4+1] * scale)) << 8) |
                   (((uint8_t)__float2int_rn(av[16+i*4+2] * scale)) << 16) |
                   (((uint8_t)__float2int_rn(av[16+i*4+3] * scale)) << 24);

        sumi = dp4a(a_lo, w_lo, sumi);
        sumi = dp4a(a_hi, w_hi, sumi);
    }

    return d_w * (d_a * (float)sumi - 8.0f * a_sum);
}

/**
 * Warp kernel: Each warp handles 4 outputs, each lane handles all K blocks for one output
 * Layout: lane 0-7 -> output 0, lane 8-15 -> output 1, etc.
 * But we do K-splitting: each lane handles K/32 blocks
 */
__global__ void __launch_bounds__(256) gemm_warp_ksplit(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    // Each warp handles 4 outputs
    constexpr int OUTPUTS_PER_WARP = 4;
    const int output_base = warp_id * OUTPUTS_PER_WARP;

    for (int m = 0; m < M; m++) {
        const float* act_row = activation + static_cast<int64_t>(m) * K;

        #pragma unroll
        for (int out_idx = 0; out_idx < OUTPUTS_PER_WARP; out_idx++) {
            const int n = output_base + out_idx;
            if (n >= N) continue;

            // Each lane handles K/32 blocks
            float sum = 0.0f;
            for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
                const uint8_t* w_block = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
                sum += dot_block_fast(w_block, act_row + kb * QK);
            }

            // Warp reduction
            sum = warp_reduce_sum(sum);

            if (lane_id == 0) {
                output[m * N + n] = sum;
            }
        }
    }
}

/**
 * Large M kernel: Simple 1C1T approach
 */
__global__ void __launch_bounds__(256) gemm_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const float* act_row = activation + static_cast<int64_t>(m) * K;
    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const uint8_t* w_block = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
        sum += dot_block_fast(w_block, act_row + kb * QK);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 64) {
        // Small M: use warp K-splitting
        // N=7168, 4 outputs per warp -> 1792 warps needed
        // Use 224 blocks of 8 warps = 1792 warps
        const int threads = 256;
        const int warps_needed = (N + 3) / 4;
        const int warps_per_block = threads / WARP_SIZE;
        const int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
        const int final_blocks = max(224, blocks);  // At least 224 blocks for full SM utilization

        gemm_warp_ksplit<<<final_blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        const int threads = 256;
        dim3 block(threads);
        dim3 grid((N + threads - 1) / threads, M);

        gemm_large_m<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM v20");
}
