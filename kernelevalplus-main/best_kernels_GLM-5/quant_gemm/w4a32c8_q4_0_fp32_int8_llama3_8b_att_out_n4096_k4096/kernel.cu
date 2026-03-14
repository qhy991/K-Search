/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel - v1
 *
 * LLaMA-3-8B Attention Output projection: N=4096, K=4096
 *
 * Q4_0 Format:
 * - Each block has 32 values quantized to 4 bits
 * - Scale is FP16 (2 bytes)
 * - Packed 4-bit values: 16 bytes (32 values)
 * - Total: 18 bytes per block
 *
 * Formula: output = activation @ (scale_w * (q_w - 8))^T
 *
 * Key optimizations:
 * 1. M=1: Warp-based approach with parallel K reduction
 * 2. Small batches: Per-warp output element computation
 * 3. Large batches: K-batch processing with shared memory tiling
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

#define QK 32
#define WARP_SIZE 32

struct __align__(2) block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // packed 4-bit values
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Unpack 16 bytes of Q4_0 to 32 int8 values
// llama.cpp ordering: byte[i] = q[i] | (q[i+16] << 4)
// Positions 0-15: low nibbles, Positions 16-31: high nibbles
__device__ __forceinline__ void unpack_q4_0(const uint8_t* qs, int* values) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t packed = qs[i];
        values[i] = (packed & 0x0F) - 8;      // low nibble -> position i
        values[i + 16] = (packed >> 4) - 8;   // high nibble -> position i+16
    }
}

// ============================================================================
// M=1 Kernel: Each warp computes one output element
// ============================================================================
__global__ void __launch_bounds__(512) gemm_q4_0_m1_warp(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int n = blockIdx.x * num_warps + warp;
    if (n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const float* act_row = activation;
    float sum = 0.0f;

    // Each lane processes a subset of K blocks
    for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
        const block_q4_0 w_block = w_blocks[n * num_blocks_k + kb];
        const float scale_w = read_half_as_float(w_block.d);
        const float* act_ptr = act_row + kb * QK;

        // Load 32 activation values using vectorized loads
        float a[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 val = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a[i * 4 + 0] = val.x;
            a[i * 4 + 1] = val.y;
            a[i * 4 + 2] = val.z;
            a[i * 4 + 3] = val.w;
        }

        // Unpack Q4_0 values
        int q[32];
        unpack_q4_0(w_block.qs, q);

        // Compute dot product
        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            block_sum += (float)q[i] * a[i];
        }

        sum += scale_w * block_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[n] = sum;
    }
}

// ============================================================================
// Small batch kernel: Each warp computes one output element
// ============================================================================
__global__ void __launch_bounds__(512) gemm_q4_0_small_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int idx = blockIdx.x * num_warps + warp;
    if (idx >= M * N) return;

    const int m = idx / N;
    const int n = idx % N;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const float* act_row = activation + (long long)m * K;
    float sum = 0.0f;

    for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
        const block_q4_0 w_block = w_blocks[n * num_blocks_k + kb];
        const float scale_w = read_half_as_float(w_block.d);
        const float* act_ptr = act_row + kb * QK;

        // Load 32 activation values
        float a[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 val = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a[i * 4 + 0] = val.x;
            a[i * 4 + 1] = val.y;
            a[i * 4 + 2] = val.z;
            a[i * 4 + 3] = val.w;
        }

        // Unpack Q4_0 values
        int q[32];
        unpack_q4_0(w_block.qs, q);

        // Compute dot product
        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            block_sum += (float)q[i] * a[i];
        }

        sum += scale_w * block_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = sum;
    }
}

// ============================================================================
// Large batch kernel with K-batch processing
// ============================================================================
constexpr int TILE_M = 8;
constexpr int TILE_N = 64;
constexpr int K_BATCH = 4;

__global__ void __launch_bounds__(512) gemm_q4_0_large_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Each warp handles multiple output elements
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int n_per_warp = TILE_N / warps_per_block;
    const int local_n = warp * n_per_warp + lane;
    const int n = block_n + local_n;

    const int m = block_m + (lane / (TILE_N / warps_per_block));
    const bool valid = (m < M) && (n < N);

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const int num_blocks_k = K / QK;

    __shared__ float s_scales[K_BATCH][TILE_N];
    __shared__ int8_t s_qs[K_BATCH][TILE_N][32];

    float sum = 0.0f;

    for (int kb_start = 0; kb_start < num_blocks_k; kb_start += K_BATCH) {
        const int kb_end = min(kb_start + K_BATCH, num_blocks_k);
        const int actual_batch = kb_end - kb_start;

        // Load weights into shared memory
        for (int k = 0; k < actual_batch; k++) {
            const int kb = kb_start + k;
            for (int load_n = tid; load_n < TILE_N; load_n += blockDim.x) {
                const int global_n = block_n + load_n;
                if (global_n < N) {
                    const block_q4_0 w_block = w_blocks[global_n * num_blocks_k + kb];
                    s_scales[k][load_n] = read_half_as_float(w_block.d);
                    // Unpack Q4_0 values into shared memory
                    const uint8_t* qs = w_block.qs;
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        s_qs[k][load_n][i] = (int8_t)((qs[i] & 0x0F) - 8);
                        s_qs[k][load_n][i + 16] = (int8_t)((qs[i] >> 4) - 8);
                    }
                }
            }
        }

        __syncthreads();

        // Compute partial results
        for (int k = 0; k < actual_batch; k++) {
            const int kb = kb_start + k;

            if (valid) {
                const float* act_ptr = activation + (long long)m * K + kb * QK;

                // Load activations
                float a[32];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float4 val = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
                    a[i * 4 + 0] = val.x;
                    a[i * 4 + 1] = val.y;
                    a[i * 4 + 2] = val.z;
                    a[i * 4 + 3] = val.w;
                }

                const float scale_w = s_scales[k][local_n];
                float block_sum = 0.0f;

                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    block_sum += (float)s_qs[k][local_n][i] * a[i];
                }

                sum += scale_w * block_sum;
            }
        }

        __syncthreads();
    }

    if (valid) {
        output[(long long)m * N + n] = sum;
    }
}

// Alternative large batch kernel with better occupancy
__global__ void __launch_bounds__(256) gemm_q4_0_large_batch_v2(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int m = blockIdx.y;
    const int n = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;

    if (m >= M || n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const float* act_row = activation + (long long)m * K;
    float sum = 0.0f;

    for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
        const block_q4_0 w_block = w_blocks[n * num_blocks_k + kb];
        const float scale_w = read_half_as_float(w_block.d);
        const float* act_ptr = act_row + kb * QK;

        // Load 32 activation values
        float a[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 val = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a[i * 4 + 0] = val.x;
            a[i * 4 + 1] = val.y;
            a[i * 4 + 2] = val.z;
            a[i * 4 + 3] = val.w;
        }

        // Unpack Q4_0 values
        int q[32];
        unpack_q4_0(w_block.qs, q);

        // Compute dot product
        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            block_sum += (float)q[i] * a[i];
        }

        sum += scale_w * block_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = sum;
    }
}

// ============================================================================
// PyTorch binding
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K)
{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(K % QK == 0, "K must be multiple of 32");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M == 1) {
        // Single token: warp-based approach
        const int threads = 512;
        const int num_warps = threads / WARP_SIZE;
        const int blocks = (N + num_warps - 1) / num_warps;

        gemm_q4_0_m1_warp<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );
    }
    else if (M <= 32) {
        // Small batch: each warp computes one output element
        const int threads = 512;
        const int num_warps = threads / WARP_SIZE;
        const int blocks = (M * N + num_warps - 1) / num_warps;

        gemm_q4_0_small_batch<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
    else {
        // Large batch: one block per row, multiple warps per block
        const int threads = 256;
        const int warps_per_block = threads / WARP_SIZE;
        dim3 block(threads);
        dim3 grid((N + warps_per_block - 1) / warps_per_block, M);

        gemm_q4_0_large_batch_v2<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM for LLaMA-3-8B Attention Output");
}
