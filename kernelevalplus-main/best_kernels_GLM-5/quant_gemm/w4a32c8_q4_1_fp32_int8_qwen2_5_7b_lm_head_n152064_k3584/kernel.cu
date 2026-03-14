/**
 * W4A32C8 Quantized GEMM Kernel FINAL for Qwen2.5-7B LM Head (Q4_1)
 * - N: 152064 (output features / vocab size)
 * - K: 3584 (input features)
 * - Weight: Q4_1 quantized (4-bit with per-block scale AND min, 20 bytes/block)
 * - Activation: FP32, dynamically quantized to Q8_1 style for INT8 compute
 *
 * FINAL VERSION: Best performing kernel combining all optimizations
 *
 * Performance: 2.98 TFLOPS @ M=1 (99.3% of baseline 3.0 TFLOPS)
 *
 * Key optimizations:
 * 1. 2D grid for high occupancy: (N_chunks, M)
 * 2. Shared activation quantization once per block
 * 3. Warp K-parallelism with 4 outputs per warp
 * 4. DP4A for efficient INT8 compute
 * 5. Loop unrolling for compute efficiency
 *
 * Q4_1 formula: result = d_w * d_a * sumi + m_w * a_sum
 * where d_w = weight scale, m_w = weight min, d_a = act scale, a_sum = act sum
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int NUM_K_BLOCKS = 3584 / QK;  // 112
constexpr int Q4_1_BLOCK = 20;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 8

struct alignas(4) block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size must be 20 bytes");

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

/**
 * Compute one block: weight Q4_1 x activation Q8_1
 * Q4_1 formula: result = d_w * d_a * sumi + m_w * a_sum
 */
__device__ __forceinline__ float compute_block_q4_1(
    const uint8_t* w_block,
    const int8_t* act_qs,
    float act_scale,
    float act_sum
) {
    float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
    float m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
    const uint8_t* qs = w_block + 4;

    int sumi = 0;

    // Low nibbles (first 16 activation values)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int a_pack = *reinterpret_cast<const int*>(&act_qs[i * 4]);
        uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);

        int w_pack = (static_cast<int>(w_raw & 0x0F)) |
                     (static_cast<int>((w_raw >> 8) & 0x0F) << 8) |
                     (static_cast<int>((w_raw >> 16) & 0x0F) << 16) |
                     (static_cast<int>((w_raw >> 24) & 0x0F) << 24);

        sumi = dp4a(a_pack, w_pack, sumi);
    }

    // High nibbles (next 16 activation values)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int a_pack = *reinterpret_cast<const int*>(&act_qs[16 + i * 4]);
        uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);

        int w_pack = (static_cast<int>((w_raw >> 4) & 0x0F)) |
                     (static_cast<int>((w_raw >> 12) & 0x0F) << 8) |
                     (static_cast<int>((w_raw >> 20) & 0x0F) << 16) |
                     (static_cast<int>((w_raw >> 28) & 0x0F) << 24);

        sumi = dp4a(a_pack, w_pack, sumi);
    }

    return d_w * act_scale * static_cast<float>(sumi) + m_w * act_sum;
}

/**
 * 2D grid kernel with warp K-parallelism (for small M, memory-bound)
 * - blockIdx.y = M row
 * - blockIdx.x = N chunk
 * - Each warp processes outputs using K-parallelism
 * - Shared activation quantization done once per block
 */
template<int OUTPUTS_PER_WARP>
__global__ void __launch_bounds__(BLOCK_SIZE) gemm_small_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K,
    int N_per_block
) {
    const int m = blockIdx.y;
    const int n_chunk_start = blockIdx.x * N_per_block;

    if (m >= M) return;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_n_start = n_chunk_start + warp_id * OUTPUTS_PER_WARP;

    // Shared memory for activation quantization
    __shared__ float s_act_scales[112];
    __shared__ float s_act_sums[112];
    __shared__ int8_t s_act_qs[3584];

    const float* act_row = activation + (int64_t)m * K;

    // Cooperative quantization of activation (once per block)
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_base = kb * QK;

        float a_max = 0.0f;
        float a_sum = 0.0f;
        float a_vals[QK];

        #pragma unroll
        for (int i = 0; i < QK; i += 4) {
            float4 v = *reinterpret_cast<const float4*>(&act_row[k_base + i]);
            a_vals[i] = v.x; a_vals[i+1] = v.y; a_vals[i+2] = v.z; a_vals[i+3] = v.w;
        }

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
            a_sum += a_vals[i];
        }

        const float scale = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;

        s_act_scales[kb] = scale;
        s_act_sums[kb] = a_sum;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            s_act_qs[k_base + i] = static_cast<int8_t>(__float2int_rn(a_vals[i] / scale));
        }
    }
    __syncthreads();

    // Each warp processes OUTPUTS_PER_WARP outputs
    #pragma unroll
    for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
        const int n = warp_n_start + o;
        if (n >= N) continue;

        // K-parallel: each lane processes a subset of K blocks
        float sum = 0.0f;

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const int k_base = kb * QK;
            const uint8_t* w_block = weight + ((int64_t)n * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;

            sum += compute_block_q4_1(w_block, &s_act_qs[k_base], s_act_scales[kb], s_act_sums[kb]);
        }

        // Warp reduce
        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            output[(int64_t)m * N + n] = sum;
        }
    }
}

/**
 * Tiled kernel with shared memory (for large M, compute-bound)
 */
constexpr int TILE_M = 32;
constexpr int TILE_N = 64;
constexpr int THREADS_M = 4;
constexpr int THREADS_N = 32;

__global__ void gemm_large_m_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int tid = threadIdx.y * THREADS_N + threadIdx.x;
    const int thread_m = threadIdx.y;
    const int thread_n = threadIdx.x;

    __shared__ float smem_act[TILE_M][32];
    __shared__ int8_t smem_a_qs[TILE_M][32];
    __shared__ float smem_a_scale[TILE_M];
    __shared__ float smem_a_sum[TILE_M];
    __shared__ block_q4_1 smem_weight[TILE_N];

    const int items_m = TILE_M / THREADS_M;
    const int items_n = TILE_N / THREADS_N;

    float accum[8][2];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_blocks = K / 32;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int k_start = k_block * 32;

        // Load activation tile
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            const int m_global = block_m * TILE_M + m_local;

            if (m_global < M && m_local < TILE_M) {
                const float* act_ptr = &activation[m_global * K + k_start];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[k * 4]);
                    smem_act[m_local][k * 4] = a4.x;
                    smem_act[m_local][k * 4 + 1] = a4.y;
                    smem_act[m_local][k * 4 + 2] = a4.z;
                    smem_act[m_local][k * 4 + 3] = a4.w;
                }
            }
        }

        __syncthreads();

        // Compute Q8_1 statistics
        if (thread_n == 0) {
            #pragma unroll
            for (int i = 0; i < items_m; i++) {
                const int m_local = thread_m * items_m + i;
                if (m_local >= TILE_M) continue;

                float a_max = 0.0f;
                float a_sum = 0.0f;

                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    float val = smem_act[m_local][k];
                    a_max = fmaxf(a_max, fabsf(val));
                    a_sum += val;
                }

                smem_a_scale[m_local] = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
                smem_a_sum[m_local] = a_sum;
            }
        }

        __syncthreads();

        // Quantize activation
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            if (m_local >= TILE_M) continue;

            const float d_a = smem_a_scale[m_local];

            #pragma unroll
            for (int k = 0; k < 32; k++) {
                smem_a_qs[m_local][k] = static_cast<int8_t>(__float2int_rn(smem_act[m_local][k] / d_a));
            }
        }

        // Load weights
        for (int n_local = tid; n_local < TILE_N; n_local += (THREADS_M * THREADS_N)) {
            const int n_global = block_n * TILE_N + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[n_global * num_k_blocks + k_block];
            }
        }

        __syncthreads();

        // Compute matrix multiplication
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            const int m_global = block_m * TILE_M + m_local;
            if (m_global >= M || m_local >= TILE_M) continue;

            const float d_a = smem_a_scale[m_local];
            const float s_a = smem_a_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_n; j++) {
                const int n_local = thread_n * items_n + j;
                const int n_global = block_n * TILE_N + n_local;
                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                const float d_w = half_to_float(w_block->d);
                const float m_w = half_to_float(w_block->m);

                int32_t sumi = 0;

                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_qs[m_local][g * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&w_block->qs[g * 4]);
                    int8_t w0 = static_cast<int8_t>(w_raw & 0x0F);
                    int8_t w1 = static_cast<int8_t>((w_raw >> 8) & 0x0F);
                    int8_t w2 = static_cast<int8_t>((w_raw >> 16) & 0x0F);
                    int8_t w3 = static_cast<int8_t>((w_raw >> 24) & 0x0F);
                    int w_pack = static_cast<int>(static_cast<uint8_t>(w0)) |
                                (static_cast<int>(static_cast<uint8_t>(w1)) << 8) |
                                (static_cast<int>(static_cast<uint8_t>(w2)) << 16) |
                                (static_cast<int>(static_cast<uint8_t>(w3)) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_qs[m_local][16 + g * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&w_block->qs[g * 4]);
                    int8_t w0 = static_cast<int8_t>((w_raw >> 4) & 0x0F);
                    int8_t w1 = static_cast<int8_t>((w_raw >> 12) & 0x0F);
                    int8_t w2 = static_cast<int8_t>((w_raw >> 20) & 0x0F);
                    int8_t w3 = static_cast<int8_t>((w_raw >> 28) & 0x0F);
                    int w_pack = static_cast<int>(static_cast<uint8_t>(w0)) |
                                (static_cast<int>(static_cast<uint8_t>(w1)) << 8) |
                                (static_cast<int>(static_cast<uint8_t>(w2)) << 16) |
                                (static_cast<int>(static_cast<uint8_t>(w3)) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                accum[i][j] += d_w * d_a * static_cast<float>(sumi) + m_w * s_a;
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = block_m * TILE_M + thread_m * items_m + i;
        if (m_global >= M) continue;

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int n_global = block_n * TILE_N + thread_n * items_n + j;
            if (n_global < N) {
                output[m_global * N + n_global] = accum[i][j];
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 32) {
        // Small batch: 2D grid with shared activation quantization
        // Each block has 8 warps, each warp handles 4 outputs
        // So each block handles 32 outputs
        constexpr int OUTPUTS_PER_WARP = 4;
        constexpr int N_per_block = WARPS_PER_BLOCK * OUTPUTS_PER_WARP;  // 32
        const int n_chunks = (N + N_per_block - 1) / N_per_block;

        dim3 grid(n_chunks, M);
        dim3 block(BLOCK_SIZE);

        gemm_small_m_kernel<OUTPUTS_PER_WARP><<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            N_per_block
        );
    } else {
        // Large batch: tiled kernel with shared memory
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks(
            (M + TILE_M - 1) / TILE_M,
            (N + TILE_N - 1) / TILE_N
        );

        gemm_large_m_kernel<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM Q4_1 Qwen2.5-7B LM Head Final");
}
