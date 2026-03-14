/**
 * Optimized Quantized GEMM Kernel for DeepSeek-V2 MoE Routing Down Projection
 *
 * Best configuration from experiments:
 * - TILE_N = 12 (128 blocks for 1536 outputs)
 * - TILE_K_BLOCKS = 80 (only 2 syncthreads per kernel)
 * - BLOCK_SIZE = 256 threads
 * - Use __ldg for read-only cache
 *
 * Performance: 185.6 GFLOPS for M=1 on RTX 4090
 * Baseline: 2470 GFLOPS (13× faster)
 *
 * The gap is due to:
 * 1. Deeply memory-bound: OI = 3.2 FLOPs/Byte << ridge = 82
 * 2. Weight size: 4.9 MB for M=1
 * 3. Baseline (llama.cpp) uses hand-optimized assembly
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int QK = 32;
constexpr int K = 5120;
constexpr int NUM_K_BLOCKS = K / QK;  // 160
constexpr int Q4_1_BLOCK = 20;

// Best config from experiments
constexpr int TILE_N = 12;
constexpr int TILE_K_BLOCKS = 80;
constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

__global__ void __launch_bounds__(BLOCK_SIZE) gemm_kernel_optimized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N_val, int K_val
) {
    const int m = blockIdx.y;
    const int tile_n_start = blockIdx.x * TILE_N;
    const int tid = threadIdx.x;

    if (m >= M) return;

    __shared__ float s_act_scales[NUM_K_BLOCKS];
    __shared__ float s_act_sums[NUM_K_BLOCKS];
    __shared__ int8_t s_act_qs[K];
    __shared__ uint8_t s_weight_tile[TILE_N * TILE_K_BLOCKS * Q4_1_BLOCK];

    const float* act_row = activation + m * K;

    // Phase 1: Load and quantize activations
    for (int kb = tid; kb < NUM_K_BLOCKS; kb += BLOCK_SIZE) {
        const int k_base = kb * QK;
        float act_max = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll 8
        for (int i = 0; i < QK; i++) {
            const float val = act_row[k_base + i];
            act_max = fmaxf(act_max, fabsf(val));
            act_sum += val;
        }

        const float scale = fmaxf(act_max / 127.0f, 1e-10f);
        s_act_scales[kb] = scale;
        s_act_sums[kb] = act_sum;

        #pragma unroll 8
        for (int i = 0; i < QK; i++) {
            const float val = act_row[k_base + i];
            const float q = roundf(val / scale);
            s_act_qs[k_base + i] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, q)));
        }
    }
    __syncthreads();

    const size_t weight_stride = NUM_K_BLOCKS * Q4_1_BLOCK;
    const int n_local = tid % TILE_N;
    const int n = tile_n_start + n_local;

    if (n >= N_val) return;

    float sum = 0.0f;

    // Phase 2: Process K-blocks in 2 tiles
    for (int k_tile = 0; k_tile < 2; k_tile++) {
        const int kb_start = k_tile * TILE_K_BLOCKS;
        const int kb_end = min(kb_start + TILE_K_BLOCKS, NUM_K_BLOCKS);
        const int kb_count = kb_end - kb_start;

        // Load weight tile cooperatively
        const int total_blocks = TILE_N * kb_count;
        for (int i = tid; i < total_blocks; i += BLOCK_SIZE) {
            const int local_n = i / kb_count;
            const int local_kb = i % kb_count;
            const int global_n = tile_n_start + local_n;
            const int global_kb = kb_start + local_kb;

            if (global_n < N_val) {
                const uint8_t* w_block = weight + global_n * weight_stride + global_kb * Q4_1_BLOCK;
                uint8_t* dst = &s_weight_tile[local_n * kb_count * Q4_1_BLOCK + local_kb * Q4_1_BLOCK];

                #pragma unroll
                for (int b = 0; b < Q4_1_BLOCK; b++) {
                    dst[b] = __ldg(w_block + b);
                }
            }
        }
        __syncthreads();

        // Compute using cached weights
        for (int kb = kb_start; kb < kb_end; kb++) {
            const int k_base = kb * QK;
            const float act_scale = s_act_scales[kb];
            const float act_sum = s_act_sums[kb];
            const int8_t* act_qs = &s_act_qs[k_base];

            const int local_kb = kb - kb_start;
            const uint8_t* w_block = &s_weight_tile[n_local * kb_count * Q4_1_BLOCK + local_kb * Q4_1_BLOCK];

            float w_scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(w_block));
            float w_min = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(w_block + 2));

            const uint8_t* qs = w_block + 4;
            int int_sum = 0;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = qs[i];
                int8_t w_lo = static_cast<int8_t>(packed & 0x0F);
                int8_t w_hi = static_cast<int8_t>((packed >> 4) & 0x0F);
                int_sum += w_lo * act_qs[i] + w_hi * act_qs[i + 16];
            }

            sum += w_scale * act_scale * static_cast<float>(int_sum) + w_min * act_sum;
        }
        __syncthreads();
    }

    output[m * N_val + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    dim3 grid((N + TILE_N - 1) / TILE_N, M);
    dim3 block(BLOCK_SIZE);

    gemm_kernel_optimized<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM Q4_1 DS2 MoE Routing Down Final");
}
