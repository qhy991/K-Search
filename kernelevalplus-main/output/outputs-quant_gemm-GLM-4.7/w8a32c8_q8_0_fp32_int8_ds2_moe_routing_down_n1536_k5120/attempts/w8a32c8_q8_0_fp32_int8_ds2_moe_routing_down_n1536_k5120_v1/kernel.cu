#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// BLOCK_Q8_0 structure: 2 bytes scale + 32 bytes int8 values = 34 bytes per block
struct __align__(2) block_q8_0 {
    uint16_t d;        // scale stored as FP16 bits
    int8_t qs[32];     // quantized int8 values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// FP16 to FP32 conversion helper
__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// INT8 dot product using DP4A (dot product accumulate 4 pairs)
// Available on Compute Capability 6.1+ (RTX 4090 is 8.9)
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c));
    return result;
}

// ============================================================================
// Kernel for M=1: Split-K approach for maximum parallelism
// One thread per output element, split K dimension across multiple blocks
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q8_0_m1_split_k(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K,
    const int split_k_slices
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int slice_id = blockIdx.y;

    if (n >= N) return;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;
    const int blocks_per_slice = (num_blocks + split_k_slices - 1) / split_k_slices;

    const int kb_start = slice_id * blocks_per_slice;
    const int kb_end = min(kb_start + blocks_per_slice, num_blocks);

    float sum = 0.0f;

    // Process assigned K blocks
    for (int kb = kb_start; kb < kb_end; kb++) {
        // Load weight block index
        const int w_block_idx = n * num_blocks + kb;

        // Load weight scale with __ldg for read-only cache
        const uint16_t w_d = __ldg(&w_blocks[w_block_idx].d);
        const float scale_w = read_half_as_float(w_d);

        // Load quantized weights
        int8_t w_qs[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            w_qs[i] = __ldg(&w_blocks[w_block_idx].qs[i]);
        }

        // Load activation block (32 values) with vectorized loads
        const float* act_ptr = activation + kb * 32;

        // Use float4 for 128-bit loads (4 floats = 16 bytes)
        float4 act_vec[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_vec[i] = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));
        }

        // Find max absolute value for activation quantization (Q8_1 style)
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            amax = fmaxf(amax, fabsf(act_vec[i].x));
            amax = fmaxf(amax, fabsf(act_vec[i].y));
            amax = fmaxf(amax, fabsf(act_vec[i].z));
            amax = fmaxf(amax, fabsf(act_vec[i].w));
        }

        const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

        // Quantize activation to int8
        int8_t act_qs[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_qs[i * 4 + 0] = __float2int_rn(act_vec[i].x * inv_scale_a);
            act_qs[i * 4 + 1] = __float2int_rn(act_vec[i].y * inv_scale_a);
            act_qs[i * 4 + 2] = __float2int_rn(act_vec[i].z * inv_scale_a);
            act_qs[i * 4 + 3] = __float2int_rn(act_vec[i].w * inv_scale_a);

            // Clamp to int8 range
            act_qs[i * 4 + 0] = max(-128, min(127, act_qs[i * 4 + 0]));
            act_qs[i * 4 + 1] = max(-128, min(127, act_qs[i * 4 + 1]));
            act_qs[i * 4 + 2] = max(-128, min(127, act_qs[i * 4 + 2]));
            act_qs[i * 4 + 3] = max(-128, min(127, act_qs[i * 4 + 3]));
        }

        // Compute dot product using DP4A (8 instructions for 32 elements)
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 4;
            // Pack 4 int8 values into int32
            int w_packed = *reinterpret_cast<const int*>(&w_qs[idx]);
            int a_packed = *reinterpret_cast<const int*>(&act_qs[idx]);
            sumi = dp4a(w_packed, a_packed, sumi);
        }

        // Apply scales and accumulate
        sum += scale_w * scale_a * (float)sumi;
    }

    // Atomic add for split-K reduction
    atomicAdd(&output[n], sum);
}

// ============================================================================
// Kernel for small batches (2 <= M <= 8): Simple row-parallel approach
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q8_0_small_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks; kb++) {
        const int w_block_idx = n * num_blocks + kb;
        const uint16_t w_d = __ldg(&w_blocks[w_block_idx].d);
        const float scale_w = read_half_as_float(w_d);

        int8_t w_qs[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            w_qs[i] = __ldg(&w_blocks[w_block_idx].qs[i]);
        }

        const float* act_ptr = activation + m * K + kb * 32;

        float4 act_vec[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_vec[i] = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));
        }

        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            amax = fmaxf(amax, fabsf(act_vec[i].x));
            amax = fmaxf(amax, fabsf(act_vec[i].y));
            amax = fmaxf(amax, fabsf(act_vec[i].z));
            amax = fmaxf(amax, fabsf(act_vec[i].w));
        }

        const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

        int8_t act_qs[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_qs[i * 4 + 0] = max(-128, min(127, __float2int_rn(act_vec[i].x * inv_scale_a)));
            act_qs[i * 4 + 1] = max(-128, min(127, __float2int_rn(act_vec[i].y * inv_scale_a)));
            act_qs[i * 4 + 2] = max(-128, min(127, __float2int_rn(act_vec[i].z * inv_scale_a)));
            act_qs[i * 4 + 3] = max(-128, min(127, __float2int_rn(act_vec[i].w * inv_scale_a)));
        }

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 4;
            int w_packed = *reinterpret_cast<const int*>(&w_qs[idx]);
            int a_packed = *reinterpret_cast<const int*>(&act_qs[idx]);
            sumi = dp4a(w_packed, a_packed, sumi);
        }

        sum += scale_w * scale_a * (float)sumi;
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Kernel for large batches (M >= 16): Tiled approach with shared memory
// ============================================================================
constexpr int TILE_M = 16;
constexpr int TILE_N = 32;
constexpr int TILE_K_BLOCKS = 4;  // Process 4 K-blocks (128 values) per iteration

__global__ void __launch_bounds__(512) gemm_q8_0_large_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    const int lane = tid & 31;       // lane ID within warp (0-31)
    const int warp = tid >> 5;       // warp ID within block (0-15)

    const int m = block_m + warp;
    const int n = block_n + lane;

    const bool valid = (m < M) && (n < N);

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;

    // Shared memory for weight blocks (scales + quantized values)
    __shared__ float s_weight_scales[TILE_K_BLOCKS][TILE_N];
    __shared__ int8_t s_weight_qs[TILE_K_BLOCKS][TILE_N][32];

    float sum = 0.0f;

    const int num_iterations = (num_blocks + TILE_K_BLOCKS - 1) / TILE_K_BLOCKS;

    for (int iter = 0; iter < num_iterations; iter++) {
        const int kb_start = iter * TILE_K_BLOCKS;
        const int kb_end = min(kb_start + TILE_K_BLOCKS, num_blocks);
        const int actual_k_blocks = kb_end - kb_start;

        // Load weight blocks into shared memory
        #pragma unroll
        for (int k = 0; k < TILE_K_BLOCKS; k++) {
            const int kb = kb_start + k;
            const int load_lane = tid & 31;

            if (k < actual_k_blocks && warp == 0) {
                const int load_n = block_n + load_lane;
                if (load_n < N) {
                    const block_q8_0 w_block = w_blocks[load_n * num_blocks + kb];
                    s_weight_scales[k][load_lane] = read_half_as_float(w_block.d);

                    #pragma unroll
                    for (int i = 0; i < 32; i++) {
                        s_weight_qs[k][load_lane][i] = w_block.qs[i];
                    }
                }
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K_BLOCKS; k++) {
            if (k >= actual_k_blocks) break;

            const int kb = kb_start + k;

            if (valid) {
                const float* act_ptr = activation + m * K + kb * 32;

                float4 act_vec[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    act_vec[i] = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));
                }

                float amax = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    amax = fmaxf(amax, fabsf(act_vec[i].x));
                    amax = fmaxf(amax, fabsf(act_vec[i].y));
                    amax = fmaxf(amax, fabsf(act_vec[i].z));
                    amax = fmaxf(amax, fabsf(act_vec[i].w));
                }

                const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

                int8_t act_qs[32];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    act_qs[i * 4 + 0] = max(-128, min(127, __float2int_rn(act_vec[i].x * inv_scale_a)));
                    act_qs[i * 4 + 1] = max(-128, min(127, __float2int_rn(act_vec[i].y * inv_scale_a)));
                    act_qs[i * 4 + 2] = max(-128, min(127, __float2int_rn(act_vec[i].z * inv_scale_a)));
                    act_qs[i * 4 + 3] = max(-128, min(127, __float2int_rn(act_vec[i].w * inv_scale_a)));
                }

                const float scale_w = s_weight_scales[k][lane];

                int sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int idx = i * 4;
                    int w_packed = *reinterpret_cast<const int*>(&s_weight_qs[k][lane][idx]);
                    int a_packed = *reinterpret_cast<const int*>(&act_qs[idx]);
                    sumi = dp4a(w_packed, a_packed, sumi);
                }

                sum += scale_w * scale_a * (float)sumi;
            }
        }

        __syncthreads();
    }

    if (valid) {
        output[m * N + n] = sum;
    }
}

// ============================================================================
// PyTorch binding with strategy dispatch
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M == 1) {
        // M=1: Use split-K for maximum parallelism
        // Split K dimension into slices for better GPU utilization
        const int num_blocks_k = K / 32;
        const int split_k = min(num_blocks_k, 128);  // Limit slices for efficiency

        dim3 block(256);
        dim3 grid((N + 255) / 256, split_k);

        gemm_q8_0_m1_split_k<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K, split_k
        );
    }
    else if (M <= 8) {
        // Small batch: Simple row-parallel
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);

        gemm_q8_0_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
    else {
        // Large batch: Tiled with shared memory
        dim3 block(512);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        gemm_q8_0_large_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W8A32C8 Q8_0 x FP32_INT8 GEMM forward pass");
}
