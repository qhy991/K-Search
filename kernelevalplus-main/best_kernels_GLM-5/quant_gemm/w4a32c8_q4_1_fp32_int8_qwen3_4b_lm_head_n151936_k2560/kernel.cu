/**
 * Qwen3-4B LM Head - Q4_1 Quantized GEMM Final
 * Combined kernel with best strategies for each M regime
 *
 * Task: C = A @ W^T where A is M×K FP32, W is N×(K/32) Q4_1 quantized
 * Dimensions: N=151936 (vocabulary), K=2560, M varies (1-512)
 *
 * Strategy:
 *   M=1: Half-warp K-parallelism (16 lanes/output, better for small K)
 *   M=2-8: Warp K-parallelism with shared activation (4 outputs/warp)
 *   M>8: Tiled kernel with shared memory
 *
 * Q4_1 Format (20 bytes per 32 values):
 *   - Bytes 0-1: delta (fp16) = (max - min) / 15
 *   - Bytes 2-3: min (fp16)
 *   - Bytes 4-19: 16 bytes of packed 4-bit UNSIGNED values [0, 15]
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK4_1 = 32;
constexpr int WARP_SIZE = 32;
constexpr int K_VAL = 2560;
constexpr int NUM_K_BLOCKS = K_VAL / QK4_1;  // 80

struct alignas(4) block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
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

__device__ __forceinline__ float half_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Shared memory structure for quantized activation blocks
struct quant_block {
    float scale;
    float sum;
    int8_t qs[QK4_1];
};

// ============================================================================
// Strategy 1: Half-warp K-parallelism (M=1, best for small K)
// ============================================================================

__global__ void __launch_bounds__(256) gemm_half_warp_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N)
{
    const int tid = threadIdx.x;
    const int lane_id = tid & 15;
    const int half_warp_id = (tid >> 4) & 1;
    const int warp_id = tid >> 5;

    __shared__ quant_block act_blocks[NUM_K_BLOCKS];

    for (int kb = tid; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_start = kb * QK4_1;
        float a_vals[QK4_1];
        #pragma unroll
        for (int i = 0; i < QK4_1; i += 4) {
            float4 v = *reinterpret_cast<const float4*>(&activation[k_start + i]);
            a_vals[i] = v.x; a_vals[i+1] = v.y; a_vals[i+2] = v.z; a_vals[i+3] = v.w;
        }
        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
            a_sum += a_vals[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        act_blocks[kb].scale = d_a;
        act_blocks[kb].sum = a_sum;
        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            act_blocks[kb].qs[i] = (int8_t)__float2int_rn(a_vals[i] / d_a);
        }
    }
    __syncthreads();

    const int global_half_warp_id = blockIdx.x * (blockDim.x / 16) + (warp_id * 2 + half_warp_id);
    const int total_half_warps = gridDim.x * (blockDim.x / 16);

    for (int n = global_half_warp_id; n < N; n += total_half_warps) {
        float sum = 0.0f;
        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += 16) {
            const uint8_t* w_block = weight + (int64_t(n) * NUM_K_BLOCKS + kb) * 20;
            const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
            const float m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
            const uint8_t* qs = w_block + 4;

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int a_pack = *reinterpret_cast<const int*>(&act_blocks[kb].qs[i * 4]);
                uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);
                int w_pack = (int(w_raw & 0x0F)) | (int((w_raw >> 8) & 0x0F) << 8) |
                            (int((w_raw >> 16) & 0x0F) << 16) | (int((w_raw >> 24) & 0x0F) << 24);
                sumi = dp4a(a_pack, w_pack, sumi);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int a_pack = *reinterpret_cast<const int*>(&act_blocks[kb].qs[16 + i * 4]);
                uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);
                int w_pack = (int((w_raw >> 4) & 0x0F)) | (int((w_raw >> 12) & 0x0F) << 8) |
                            (int((w_raw >> 20) & 0x0F) << 16) | (int((w_raw >> 28) & 0x0F) << 24);
                sumi = dp4a(a_pack, w_pack, sumi);
            }
            sum += d_w * act_blocks[kb].scale * (float)sumi + m_w * act_blocks[kb].sum;
        }
        sum = half_warp_reduce_sum(sum);
        if (lane_id == 0) output[n] = sum;
    }
}

// ============================================================================
// Strategy 2: Warp K-parallelism (M=2-8, shared activation)
// ============================================================================

__global__ void __launch_bounds__(256) gemm_warp_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K)
{
    const int row = blockIdx.y;
    if (row >= M) return;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    __shared__ quant_block act_blocks[NUM_K_BLOCKS];

    const float* row_act = activation + int64_t(row) * K;
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_start = kb * QK4_1;
        float a_vals[QK4_1];
        #pragma unroll
        for (int i = 0; i < QK4_1; i += 4) {
            float4 v = *reinterpret_cast<const float4*>(&row_act[k_start + i]);
            a_vals[i] = v.x; a_vals[i+1] = v.y; a_vals[i+2] = v.z; a_vals[i+3] = v.w;
        }
        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
            a_sum += a_vals[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        act_blocks[kb].scale = d_a;
        act_blocks[kb].sum = a_sum;
        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            act_blocks[kb].qs[i] = (int8_t)__float2int_rn(a_vals[i] / d_a);
        }
    }
    __syncthreads();

    constexpr int OUTPUTS_PER_WARP = 4;
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    const int total_warps = gridDim.x * (blockDim.x / WARP_SIZE);

    for (int n_base = global_warp_id * OUTPUTS_PER_WARP; n_base < N; n_base += total_warps * OUTPUTS_PER_WARP) {
        float sums[4] = {0.0f};
        int n_vals[4];
        bool valid[4];
        #pragma unroll
        for (int o = 0; o < 4; o++) {
            n_vals[o] = n_base + o;
            valid[o] = (n_vals[o] < N);
        }

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            #pragma unroll
            for (int o = 0; o < 4; o++) {
                if (!valid[o]) continue;
                const uint8_t* w_block = weight + (int64_t(n_vals[o]) * NUM_K_BLOCKS + kb) * 20;
                const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
                const float m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
                const uint8_t* qs = w_block + 4;

                int sumi = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int a_pack = *reinterpret_cast<const int*>(&act_blocks[kb].qs[i * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);
                    int w_pack = (int(w_raw & 0x0F)) | (int((w_raw >> 8) & 0x0F) << 8) |
                                (int((w_raw >> 16) & 0x0F) << 16) | (int((w_raw >> 24) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int a_pack = *reinterpret_cast<const int*>(&act_blocks[kb].qs[16 + i * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&qs[i * 4]);
                    int w_pack = (int((w_raw >> 4) & 0x0F)) | (int((w_raw >> 12) & 0x0F) << 8) |
                                (int((w_raw >> 20) & 0x0F) << 16) | (int((w_raw >> 28) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }
                sums[o] += d_w * act_blocks[kb].scale * (float)sumi + m_w * act_blocks[kb].sum;
            }
        }

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            if (!valid[o]) continue;
            sums[o] = warp_reduce_sum(sums[o]);
            if (lane_id == 0) output[int64_t(row) * N + n_vals[o]] = sums[o];
        }
    }
}

// ============================================================================
// Strategy 3: Tiled kernel (M > 8)
// ============================================================================

constexpr int TILE_M = 16;
constexpr int TILE_N = 128;
constexpr int THREADS_M = 4;
constexpr int THREADS_N = 32;

__global__ void __launch_bounds__(128) gemm_tiled_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K)
{
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

    float accum[4][4] = {{{0.0f}}};

    const int num_k_blocks = K / 32;
    const int m_global_base = block_m * TILE_M;
    const int n_global_base = block_n * TILE_N;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int k_start = k_block * 32;

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

        if (thread_n == 0) {
            #pragma unroll
            for (int i = 0; i < items_m; i++) {
                const int m_local = thread_m * items_m + i;
                if (m_local >= TILE_M) continue;
                float a_max = 0.0f, a_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    a_max = fmaxf(a_max, fabsf(smem_act[m_local][k]));
                    a_sum += smem_act[m_local][k];
                }
                smem_a_scale[m_local] = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
                smem_a_sum[m_local] = a_sum;
            }
        }
        __syncthreads();

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

        for (int n_local = tid; n_local < TILE_N; n_local += (THREADS_M * THREADS_N)) {
            const int n_global = n_global_base + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[int64_t(n_global) * num_k_blocks + k_block];
            }
        }
        __syncthreads();

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
                    int w_pack = (int)(uint8_t)(w_raw & 0x0F) | ((int)(uint8_t)((w_raw >> 8) & 0x0F) << 8) |
                                ((int)(uint8_t)((w_raw >> 16) & 0x0F) << 16) | ((int)(uint8_t)((w_raw >> 24) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_a_qs[m_local][16 + g * 4]);
                    uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&w_block->qs[g * 4]);
                    int w_pack = (int)(uint8_t)((w_raw >> 4) & 0x0F) | ((int)(uint8_t)((w_raw >> 12) & 0x0F) << 8) |
                                ((int)(uint8_t)((w_raw >> 20) & 0x0F) << 16) | ((int)(uint8_t)((w_raw >> 28) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }
                accum[i][j] += d_w * d_a * (float)sumi + m_w * s_a;
            }
        }
        __syncthreads();
    }

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

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        // M=1: Half-warp K-parallelism
        const int blocks = 512;
        gemm_half_warp_kernel<<<blocks, 256>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );
    } else if (M <= 8) {
        // M=2-8: Warp K-parallelism with shared activation
        dim3 grid(512, M);
        gemm_warp_shared_kernel<<<grid, 256>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // M>8: Tiled kernel
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        gemm_tiled_kernel<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Qwen3-4B LM Head Q4_1 GEMM Final");
}
