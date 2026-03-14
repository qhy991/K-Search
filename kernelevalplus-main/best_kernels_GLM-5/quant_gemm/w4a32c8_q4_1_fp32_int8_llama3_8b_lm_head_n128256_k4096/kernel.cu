/**
 * LLaMA-3-8B LM Head - Q4_1 Quantized GEMM Final
 *
 * Task: C = A @ W^T where A is M×K FP32, W is N×(K/32) Q4_1 quantized
 * Dimensions: N=128256 (vocabulary), K=4096, M varies (1-512)
 *
 * Q4_1 Format (20 bytes per 32 values):
 *   - Bytes 0-1: delta (fp16) = (max - min) / 15
 *   - Bytes 2-3: min (fp16)
 *   - Bytes 4-19: 16 bytes of packed 4-bit UNSIGNED values [0, 15]
 *   Dequantization: val = q * delta + min, where q in [0, 15]
 *
 * Roofline Analysis (RTX 4090):
 *   - Ridge Point: 81.9 FLOPs/Byte
 *   - M=1-8:   OI = 3.2-25.3 → MEMORY-BOUND (weight reuse: 3x-26x)
 *   - M=512:   OI = 897 → COMPUTE-BOUND (weight reuse: 1638x)
 *
 * Combined Strategy:
 *   - M=1-8:   Warp-level K-parallelism (v5: 2340-2668 GFLOPS)
 *   - M>8:     Tiled kernel with shared memory (v5: 9677 GFLOPS)
 *
 * Performance (RTX 4090):
 *   M=1:   ~2344 GFLOPS (warp kernel, 78% of baseline)
 *   M=2-8: ~2350-2668 GFLOPS (warp kernel)
 *   M=512: ~9708 GFLOPS (tiled kernel)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK4_1 = 32;
constexpr int WARP_SIZE = 32;
constexpr int K_VAL = 4096;
constexpr int NUM_K_BLOCKS = K_VAL / QK4_1;  // 128

struct alignas(4) block_q4_1 {
    uint16_t d;  // delta = (max - min) / 15
    uint16_t m;  // min
    uint8_t qs[16];  // packed 4-bit values
};
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size must be 20 bytes");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
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

// ============================================================================
// Strategy 1: Warp-level K-parallelism (M <= 8, memory-bound)
// Each warp processes 4 outputs, lanes collaborate on K dimension
// ============================================================================

__global__ void __launch_bounds__(512) gemm_warp_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    const int total_warps = gridDim.x * warps_per_block;

    constexpr int OUTPUTS_PER_WARP = 4;

    for (int base = global_warp_id * OUTPUTS_PER_WARP; base < M * N; base += total_warps * OUTPUTS_PER_WARP) {
        int m_idx[OUTPUTS_PER_WARP], n_idx[OUTPUTS_PER_WARP];
        float sums[OUTPUTS_PER_WARP] = {0.0f};

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            int idx = base + o;
            m_idx[o] = idx / N;
            n_idx[o] = idx % N;
            if (idx >= M * N) m_idx[o] = -1;
        }

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            int k_start = kb * QK4_1;

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (m_idx[o] < 0) continue;

                const uint8_t* w_block = weight + (int64_t(n_idx[o]) * NUM_K_BLOCKS + kb) * 20;
                const float* a_ptr = activation + m_idx[o] * K + k_start;

                float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
                float m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));

                float a_vals[QK4_1];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float4 v = *reinterpret_cast<const float4*>(a_ptr + i * 4);
                    a_vals[i * 4] = v.x;
                    a_vals[i * 4 + 1] = v.y;
                    a_vals[i * 4 + 2] = v.z;
                    a_vals[i * 4 + 3] = v.w;
                }

                float a_max = 0.0f;
                float a_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < QK4_1; i++) {
                    a_max = fmaxf(a_max, fabsf(a_vals[i]));
                    a_sum += a_vals[i];
                }

                float d_a = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;

                int8_t a_qs[QK4_1];
                #pragma unroll
                for (int i = 0; i < QK4_1; i++) {
                    a_qs[i] = static_cast<int8_t>(__float2int_rn(a_vals[i] / d_a));
                }

                const uint8_t* qs = w_block + 4;
                int sumi = 0;

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int a_pack = *reinterpret_cast<const int*>(&a_qs[i * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);
                    int w_pack = (static_cast<int>(w_raw & 0x0F)) |
                                 (static_cast<int>((w_raw >> 8) & 0x0F) << 8) |
                                 (static_cast<int>((w_raw >> 16) & 0x0F) << 16) |
                                 (static_cast<int>((w_raw >> 24) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int a_pack = *reinterpret_cast<const int*>(&a_qs[16 + i * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);
                    int w_pack = (static_cast<int>((w_raw >> 4) & 0x0F)) |
                                 (static_cast<int>((w_raw >> 12) & 0x0F) << 8) |
                                 (static_cast<int>((w_raw >> 20) & 0x0F) << 16) |
                                 (static_cast<int>((w_raw >> 28) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                sums[o] += d_w * d_a * static_cast<float>(sumi) + m_w * a_sum;
            }
        }

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            sums[o] = warp_reduce_sum(sums[o]);
            if (lane_id == 0 && m_idx[o] >= 0) {
                output[m_idx[o] * N + n_idx[o]] = sums[o];
            }
        }
    }
}

// ============================================================================
// Strategy 2: Tiled kernel (M > 8, compute-bound)
// ============================================================================

constexpr int TILE_M = 16;
constexpr int TILE_N = 128;
constexpr int THREADS_M = 4;
constexpr int THREADS_N = 32;

__global__ void __launch_bounds__(128) gemm_tiled_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;
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

    float accum[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_blocks = K / 32;
    const int m_global_base = block_m * TILE_M;
    const int n_global_base = block_n * TILE_N;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int k_start = k_block * 32;

        // Load activation tile
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            const int m_global = m_global_base + m_local;

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
                smem_a_qs[m_local][k] = (int8_t)__float2int_rn(smem_act[m_local][k] / d_a);
            }
        }

        // Load weights
        for (int n_local = tid; n_local < TILE_N; n_local += (THREADS_M * THREADS_N)) {
            const int n_global = n_global_base + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[(int64_t)n_global * num_k_blocks + k_block];
            }
        }

        __syncthreads();

        // Compute matrix multiplication
        #pragma unroll
        for (int i = 0; i < items_m; i++) {
            const int m_local = thread_m * items_m + i;
            const int m_global = m_global_base + m_local;
            if (m_global >= M || m_local >= TILE_M) continue;

            const float d_a = smem_a_scale[m_local];
            const float s_a = smem_a_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_n; j++) {
                const int n_local = thread_n * items_n + j;
                const int n_global = n_global_base + n_local;
                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                const float d_w = half_to_float(w_block->d);
                const float m_w = half_to_float(w_block->m);

                int32_t sumi = 0;

                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_qs[m_local][g * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&w_block->qs[g * 4]);
                    int8_t w0 = (int8_t)(w_raw & 0x0F);
                    int8_t w1 = (int8_t)((w_raw >> 8) & 0x0F);
                    int8_t w2 = (int8_t)((w_raw >> 16) & 0x0F);
                    int8_t w3 = (int8_t)((w_raw >> 24) & 0x0F);
                    int w_pack = (int)(uint8_t)w0 | ((int)(uint8_t)w1 << 8) |
                                ((int)(uint8_t)w2 << 16) | ((int)(uint8_t)w3 << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_qs[m_local][16 + g * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&w_block->qs[g * 4]);
                    int8_t w0 = (int8_t)((w_raw >> 4) & 0x0F);
                    int8_t w1 = (int8_t)((w_raw >> 12) & 0x0F);
                    int8_t w2 = (int8_t)((w_raw >> 20) & 0x0F);
                    int8_t w3 = (int8_t)((w_raw >> 28) & 0x0F);
                    int w_pack = (int)(uint8_t)w0 | ((int)(uint8_t)w1 << 8) |
                                ((int)(uint8_t)w2 << 16) | ((int)(uint8_t)w3 << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                accum[i][j] += d_w * d_a * (float)sumi + m_w * s_a;
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < items_m; i++) {
        const int m_global = m_global_base + thread_m * items_m + i;
        if (m_global >= M) continue;

        #pragma unroll
        for (int j = 0; j < items_n; j++) {
            const int n_global = n_global_base + thread_n * items_n + j;
            if (n_global < N) {
                output[m_global * N + n_global] = accum[i][j];
            }
        }
    }
}

// PyTorch Interface
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

    if (M <= 8) {
        // Memory-bound regime: warp-level K-parallelism
        const int total_outputs = M * N;
        constexpr int OUTPUTS_PER_WARP = 4;
        constexpr int WARPS_PER_BLOCK = 16;  // 512 threads
        const int warps_needed = (total_outputs + OUTPUTS_PER_WARP - 1) / OUTPUTS_PER_WARP;
        const int blocks = min((warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 8192);

        gemm_warp_kernel<<<blocks, 512>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Compute-bound regime: tiled kernel
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M
        );

        gemm_tiled_kernel<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LLaMA3-8B LM Head Q4_1 GEMM Final");
}
