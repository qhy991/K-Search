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

// INT8 dot product using DP4A (Tensor Core instruction)
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c));
    return result;
}

// ============================================================================
// M=1 Split-K kernel: Each thread computes partial result for one output element
// K dimension is split across multiple blocks, results combined with atomic add
// This is optimal for memory-bound M=1 case
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q8_0_m1_split_k(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K,
    const int split_k
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int split_id = blockIdx.y;

    if (n >= N) return;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;
    const int blocks_per_split = (num_blocks + split_k - 1) / split_k;

    const int kb_start = split_id * blocks_per_split;
    const int kb_end = min(kb_start + blocks_per_split, num_blocks);

    float sum = 0.0f;

    for (int kb = kb_start; kb < kb_end; kb++) {
        // Copy entire block to avoid alignment issues
        const block_q8_0 w_block = w_blocks[n * num_blocks + kb];
        const float scale_w = read_half_as_float(w_block.d);

        const float* act_ptr = activation + kb * 32;

        // Load activation block with vectorized loads (read-only cache)
        float4 act_vec[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_vec[i] = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));
        }

        // Find max for activation quantization (Q8_1 style)
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

        // Quantize activation
        int8_t act_qs[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_qs[i * 4 + 0] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].x * inv_scale_a)));
            act_qs[i * 4 + 1] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].y * inv_scale_a)));
            act_qs[i * 4 + 2] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].z * inv_scale_a)));
            act_qs[i * 4 + 3] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].w * inv_scale_a)));
        }

        // Dot product using DP4A
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 4;
            int w_packed = *reinterpret_cast<const int*>(&w_block.qs[idx]);
            int a_packed = *reinterpret_cast<int*>(&act_qs[idx]);
            sumi = dp4a(w_packed, a_packed, sumi);
        }

        sum += scale_w * scale_a * (float)sumi;
    }

    atomicAdd(&output[n], sum);
}

// ============================================================================
// Small batch kernel (2 <= M < 16): 2D block for better coalescing
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q8_0_small_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || m >= M) return;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks; kb++) {
        const block_q8_0 w_block = w_blocks[n * num_blocks + kb];
        const float scale_w = read_half_as_float(w_block.d);

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
            act_qs[i * 4 + 0] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].x * inv_scale_a)));
            act_qs[i * 4 + 1] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].y * inv_scale_a)));
            act_qs[i * 4 + 2] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].z * inv_scale_a)));
            act_qs[i * 4 + 3] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].w * inv_scale_a)));
        }

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 4;
            int w_packed = *reinterpret_cast<const int*>(&w_block.qs[idx]);
            int a_packed = *reinterpret_cast<int*>(&act_qs[idx]);
            sumi = dp4a(w_packed, a_packed, sumi);
        }

        sum += scale_w * scale_a * (float)sumi;
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Large batch tiled kernel (M >= 16): Tiled with shared memory for efficiency
// ============================================================================
constexpr int TILE_M = 16;
constexpr int TILE_N = 32;
constexpr int K_BATCH = 8;  // Process 8 K-blocks per iteration

__global__ void __launch_bounds__(512) gemm_q8_0_large_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int m = block_m + warp;
    const int n = block_n + lane;

    const bool valid = (m < M) && (n < N);

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;

    // Double buffer for weights
    __shared__ float weight_scales[2][K_BATCH][TILE_N];
    __shared__ int8_t weight_qs[2][K_BATCH][TILE_N][32];

    float sum = 0.0f;

    const int num_batches = (num_blocks + K_BATCH - 1) / K_BATCH;

    for (int batch = 0; batch < num_batches; batch++) {
        const int kb_start = batch * K_BATCH;
        const int kb_end = min(kb_start + K_BATCH, num_blocks);
        const int actual_batch = kb_end - kb_start;

        const int buf_idx = batch & 1;

        // Load weight blocks into shared memory
        for (int k = 0; k < actual_batch; k++) {
            const int kb = kb_start + k;
            const int load_col = lane;
            const int load_n = block_n + load_col;

            if (load_n < N && warp < 1) {
                const block_q8_0 w_block = w_blocks[load_n * num_blocks + kb];
                weight_scales[buf_idx][k][load_col] = read_half_as_float(w_block.d);

                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    weight_qs[buf_idx][k][load_col][i] = w_block.qs[i];
                }
            }
        }

        __syncthreads();

        // Compute
        for (int k = 0; k < actual_batch; k++) {
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

                const float scale_w = weight_scales[buf_idx][k][lane];

                int8_t act_qs[32];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    act_qs[i * 4 + 0] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].x * inv_scale_a)));
                    act_qs[i * 4 + 1] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].y * inv_scale_a)));
                    act_qs[i * 4 + 2] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].z * inv_scale_a)));
                    act_qs[i * 4 + 3] = (int8_t)max(-128, min(127, __float2int_rn(act_vec[i].w * inv_scale_a)));
                }

                int sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int idx = i * 4;
                    int w_packed = *reinterpret_cast<const int*>(&weight_qs[buf_idx][k][lane][idx]);
                    int a_packed = *reinterpret_cast<int*>(&act_qs[idx]);
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
// PyTorch binding with strategy dispatch based on M
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
        // M=1: Split-K for maximum parallelism
        // split_k = min(K/32 blocks, 256) for optimal GPU utilization
        const int num_blocks_k = K / 32;
        const int split_k = min(num_blocks_k, 256);
        dim3 block(256);
        dim3 grid((N + 255) / 256, split_k);

        gemm_q8_0_m1_split_k<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K, split_k
        );
    }
    else if (M < 16) {
        // Small batch (2-15): 2D block for better memory coalescing
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (M + 3) / 4);

        gemm_q8_0_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
    else {
        // Large batch (>=16): Tiled with shared memory
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
    m.def("forward", &forward, "W8A32C8 Q8_0 x FP32_INT8 (Q8_1-style) GEMM forward pass");
}
