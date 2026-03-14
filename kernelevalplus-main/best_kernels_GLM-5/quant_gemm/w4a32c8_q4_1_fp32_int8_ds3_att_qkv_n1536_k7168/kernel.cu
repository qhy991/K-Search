/**
 * Quantized GEMM for Deepseek-V3 Attention QKV Projection with Q4_1 Weights
 *
 * Optimized v7 with strategy dispatch based on batch size:
 *   - Small M (1-16): Coalesced memory access kernel optimized for memory-bound regime
 *   - Large M (17+): DP4A-optimized shared memory tiling for compute-bound regime
 *
 * Parameters:
 *   - N = 1536 (output dimension)
 *   - K = 7168 (hidden dimension)
 *   - M = batch size (variable, 1-512)
 *   - Weight: Q4_1 quantized (4-bit packed, stored as uint8)
 *   - Activation: FP32
 *   - Output: FP32
 *
 * Q4_1 Format (20 bytes per 32 values):
 *   - half d (2 bytes): scale factor
 *   - half m (2 bytes): minimum value (offset)
 *   - uint8_t qs[16] (16 bytes): 32 packed 4-bit values
 *
 * Dequantization: val = q * d + m
 *
 * Hardware: RTX 4090
 *   - Compute Capability: 8.9
 *   - SM Count: 128
 *   - Memory Bandwidth: ~1008 GB/s
 *   - Peak FP32 TFLOPS: ~82.6
 *   - Peak INT8 TFLOPS: ~330 (with DP4A)
 *
 * Roofline Analysis:
 *   - M=1:  OI ≈ 1.8 FLOPs/Byte → MEMORY-BOUND (bandwidth optimization)
 *   - M=512: OI ≈ 418 FLOPs/Byte → COMPUTE-BOUND (DP4A optimization)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cstdint>

// Block size constants
constexpr int QK4_1 = 32;
constexpr int QK8_1 = 32;
constexpr int WARP_SIZE = 32;

// Strategy dispatch threshold
constexpr int SMALL_BATCH_THRESHOLD = 16;

// ============================================================================
// Q4_1 Block Structure (20 bytes total)
// Layout: [FP16 scale d (2)][FP16 min m (2)][16 bytes packed 4-bit values]
// ============================================================================
struct block_q4_1 {
    uint16_t d;        // scale stored as uint16_t (raw FP16 bits)
    uint16_t m;        // minimum stored as uint16_t (raw FP16 bits)
    uint8_t qs[16];    // packed 4-bit values (32 values total, 2 per byte)
};
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size must be 20 bytes");

// FP16 to FP32 conversion
__device__ __forceinline__ float half_to_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

// DP4A instruction for INT8 dot product (CC >= 6.1)
#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a_device(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
__device__ __forceinline__ int dp4a_device(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

// ============================================================================
// Kernel 1: Small Batch - One warp per output element, optimized for memory
// Uses coalesced memory access and minimizes redundant loads
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_1_small_batch(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    const int num_blocks = K / QK4_1;  // 224 blocks

    // Each warp processes one (m, n) output element
    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int row = idx / N;
        const int col = idx % N;

        float sum = 0.0f;
        const float* act_row = activation + row * K;

        // Each lane processes a subset of K blocks
        for (int b = lane_id; b < num_blocks; b += WARP_SIZE) {
            const block_q4_1* w_block = &weight[col * num_blocks + b];

            // Load weight scales
            const float d_w = half_to_float(w_block->d);
            const float m_w = half_to_float(w_block->m);

            const int k_start = b * QK4_1;

            // Process 32 values - vectorized load of activation
            float a_local[32];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 a4 = *reinterpret_cast<const float4*>(&act_row[k_start + i * 4]);
                a_local[i * 4 + 0] = a4.x;
                a_local[i * 4 + 1] = a4.y;
                a_local[i * 4 + 2] = a4.z;
                a_local[i * 4 + 3] = a4.w;
            }

            // Compute dot product with weight dequantization
            // Q4_1: val = q * d + m, where q is 0-15
            float block_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t packed = w_block->qs[i];
                const int w0 = packed & 0x0F;
                const int w1 = (packed >> 4) & 0x0F;

                // Dequantized weight: w = d_w * q + m_w
                const float w_deq0 = d_w * w0 + m_w;
                const float w_deq1 = d_w * w1 + m_w;

                block_sum += a_local[i] * w_deq0;
                block_sum += a_local[i + 16] * w_deq1;
            }
            sum += block_sum;
        }

        // Warp reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            output[row * N + col] = sum;
        }
    }
}

// ============================================================================
// Kernel 2: Large Batch - DP4A-optimized with shared memory tiling
// ============================================================================
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;
constexpr int THREADS_M = 8;
constexpr int THREADS_N = 32;

__global__ void __launch_bounds__(256) gemm_q4_1_large_batch(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;

    const int thread_m = threadIdx.y;
    const int thread_n = threadIdx.x;
    const int tid = thread_m * THREADS_N + thread_n;

    // Shared memory for activation tile
    __shared__ float smem_act[TILE_M][TILE_K];
    __shared__ int8_t smem_act_q[TILE_M][TILE_K];
    __shared__ float smem_act_scale[TILE_M];
    __shared__ float smem_act_sum[TILE_M];
    __shared__ block_q4_1 smem_weight[TILE_N];

    const int items_per_thread_m = TILE_M / THREADS_M;
    const int items_per_thread_n = TILE_N / THREADS_N;

    float accum[items_per_thread_m][items_per_thread_n];
    #pragma unroll
    for (int i = 0; i < items_per_thread_m; ++i) {
        #pragma unroll
        for (int j = 0; j < items_per_thread_n; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_blocks = K / QK4_1;

    // Loop over K blocks
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * QK4_1;

        // Cooperatively load activation tile
        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_M * THREADS_N) {
            const int m_local = idx / TILE_K;
            const int k_local = idx % TILE_K;
            const int m_global = block_m * TILE_M + m_local;
            const int k_global = k_start + k_local;

            if (m_global < M) {
                smem_act[m_local][k_local] = activation[m_global * K + k_global];
            } else {
                smem_act[m_local][k_local] = 0.0f;
            }
        }

        __syncthreads();

        // Compute per-row quantization scale for this tile
        for (int m_base = 0; m_base < TILE_M; m_base += THREADS_M) {
            const int m_local = m_base + thread_m;
            if (m_local >= TILE_M) continue;

            float local_max = 0.0f;
            float local_sum = 0.0f;

            for (int k = thread_n; k < TILE_K; k += THREADS_N) {
                const float val = smem_act[m_local][k];
                local_max = fmaxf(local_max, fabsf(val));
                local_sum += val;
            }

            // Reduce within warp
            #pragma unroll
            for (int offset = THREADS_N / 2; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            }

            if (thread_n == 0) {
                const float d_a = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
                smem_act_scale[m_local] = d_a;
                smem_act_sum[m_local] = local_sum;
            }
        }

        __syncthreads();

        // Quantize activation to INT8
        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_M * THREADS_N) {
            const int m_local = idx / TILE_K;
            const int k_local = idx % TILE_K;
            const float val = smem_act[m_local][k_local];
            const float d_a = smem_act_scale[m_local];
            smem_act_q[m_local][k_local] = (int8_t)__float2int_rn(val / d_a);
        }

        // Load weight blocks for this N tile
        for (int n_local = tid; n_local < TILE_N; n_local += THREADS_M * THREADS_N) {
            const int n_global = block_n * TILE_N + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[n_global * num_k_blocks + k_block];
            }
        }

        __syncthreads();

        // Compute partial products using DP4A
        #pragma unroll
        for (int i = 0; i < items_per_thread_m; ++i) {
            const int m_local = thread_m * items_per_thread_m + i;
            const int m_global = block_m * TILE_M + m_local;
            if (m_global >= M) continue;

            const float d_a = smem_act_scale[m_local];
            const float s_a = smem_act_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_per_thread_n; ++j) {
                const int n_local = thread_n * items_per_thread_n + j;
                const int n_global = block_n * TILE_N + n_local;
                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                const float d_w = half_to_float(w_block->d);
                const float m_w = half_to_float(w_block->m);

                // DP4A computation
                int32_t sumi = 0;

                // First 16 values (low nibbles)
                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int a_pack = *reinterpret_cast<const int*>(&smem_act_q[m_local][ii * 4]);
                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);

                    // Unpack 4-bit weights (low nibbles)
                    const int w0 = w_packed & 0x0F;
                    const int w1 = (w_packed >> 8) & 0x0F;
                    const int w2 = (w_packed >> 16) & 0x0F;
                    const int w3 = (w_packed >> 24) & 0x0F;

                    const int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) |
                                       ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);

                    sumi = dp4a_device(a_pack, w_pack, sumi);
                }

                // Second 16 values (high nibbles)
                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int a_pack = *reinterpret_cast<const int*>(&smem_act_q[m_local][16 + ii * 4]);
                    const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);

                    // Unpack 4-bit weights (high nibbles)
                    const int w0 = (w_packed >> 4) & 0x0F;
                    const int w1 = (w_packed >> 12) & 0x0F;
                    const int w2 = (w_packed >> 20) & 0x0F;
                    const int w3 = (w_packed >> 28) & 0x0F;

                    const int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) |
                                       ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);

                    sumi = dp4a_device(a_pack, w_pack, sumi);
                }

                accum[i][j] += d_w * d_a * (float)sumi + m_w * s_a;
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < items_per_thread_m; ++i) {
        const int m_global = block_m * TILE_M + thread_m * items_per_thread_m + i;
        if (m_global >= M) continue;

        #pragma unroll
        for (int j = 0; j < items_per_thread_n; ++j) {
            const int n_global = block_n * TILE_N + thread_n * items_per_thread_n + j;
            if (n_global < N) {
                output[m_global * N + n_global] = accum[i][j];
            }
        }
    }
}

// ============================================================================
// PyTorch Interface with Strategy Dispatch
// ============================================================================

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const block_q4_1* weight_ptr = reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>());

    if (M <= SMALL_BATCH_THRESHOLD) {
        // Small batch: use memory-optimized kernel
        // Each warp processes one output element
        const int total_outputs = M * N;
        const int threads = 256;
        const int warps_per_block = threads / WARP_SIZE;
        const int num_blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

        gemm_q4_1_small_batch<<<num_blocks, threads>>>(
            weight_ptr,
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: use DP4A-optimized tiled kernel
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks(
            (M + TILE_M - 1) / TILE_M,
            (N + TILE_N - 1) / TILE_N
        );

        gemm_q4_1_large_batch<<<blocks, threads>>>(
            weight_ptr,
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM for Deepseek-V3 Attention QKV Projection");
}
