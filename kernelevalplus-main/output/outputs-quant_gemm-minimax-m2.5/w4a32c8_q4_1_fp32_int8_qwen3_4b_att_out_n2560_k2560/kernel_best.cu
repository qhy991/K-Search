// W4A32C8 Q4_1 Quantized GEMM for Qwen3-4B Attention Output - v3
// M: variable (1-512), N: 2560, K: 2560
// Optimized: Simple kernel for M=1, tiled for M>=2

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define QK4_1 32
#define WARP_SIZE 32
#define K_BLOCKS 80

// Q4_1 block: 20 bytes
struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

inline __device__ float half2float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

#if __CUDA_ARCH__ >= 610
inline __device__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
inline __device__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

// ============================================================================
// Kernel 1: Simple row kernel for M=1
// ============================================================================
__global__ void w4a32c8_q4_1_kernel_simple(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    const int num_k_blocks = K / QK4_1;

    if (tid >= total) return;

    const int row = tid / N;
    const int col = tid % N;

    const float* a_ptr = activation + row * K;
    const block_q4_1* w_ptr = weight + col * num_k_blocks;

    float sum = 0.0f;

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        const block_q4_1& wb = w_ptr[kb];
        const float* a_block = a_ptr + kb * QK4_1;

        float d_w = half2float(wb.d);
        float m_w = half2float(wb.m);
        if (d_w == 0.0f) d_w = 1.0f;

        float a_max = 0.0f;
        float a_sum = 0.0f;

        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            float4 av = *reinterpret_cast<const float4*>(a_block + i * 4);
            a_max = fmaxf(a_max, fmaxf(fabsf(av.x), fmaxf(fabsf(av.y), fmaxf(fabsf(av.z), fabsf(av.w)))));
            a_sum += av.x + av.y + av.z + av.w;
        }

        float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_pack[8];
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            float4 av = *reinterpret_cast<const float4*>(a_block + i * 4);
            int8_t q0 = (int8_t)__float2int_rn(av.x / d_a);
            int8_t q1 = (int8_t)__float2int_rn(av.y / d_a);
            int8_t q2 = (int8_t)__float2int_rn(av.z / d_a);
            int8_t q3 = (int8_t)__float2int_rn(av.w / d_a);
            a_pack[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                        ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        int32_t sumi = 0;
        #pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            uint32_t w_p = *reinterpret_cast<const uint32_t*>(&wb.qs[i * 4]);
            int w0 = (int)(w_p & 0x0F);
            int w1 = (int)((w_p >> 8) & 0x0F);
            int w2 = (int)((w_p >> 16) & 0x0F);
            int w3 = (int)((w_p >> 24) & 0x0F);
            int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
            sumi = dp4a(a_pack[i], w_pack, sumi);
        }
        #pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            uint32_t w_p = *reinterpret_cast<const uint32_t*>(&wb.qs[i * 4]);
            int w0 = (int)((w_p >> 4) & 0x0F);
            int w1 = (int)((w_p >> 12) & 0x0F);
            int w2 = (int)((w_p >> 20) & 0x0F);
            int w3 = (int)((w_p >> 28) & 0x0F);
            int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
            sumi = dp4a(a_pack[i + 4], w_pack, sumi);
        }

        sum += d_w * d_a * (float)sumi + m_w * a_sum;
    }

    output[row * N + col] = sum;
}

// ============================================================================
// Kernel 2: Tiled kernel for M>=2
// ============================================================================
#define TILE_M 64
#define TILE_N 64
#define THREADS_M 8
#define THREADS_N 16

__global__ void w4a32c8_q4_1_kernel_tiled_opt(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;

    const int tid = threadIdx.y * THREADS_N + threadIdx.x;
    const int thread_m = threadIdx.y;
    const int thread_n = threadIdx.x;

    __shared__ float smem_act[TILE_M][QK4_1];
    __shared__ int8_t smem_act_q[TILE_M][QK4_1];
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

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int act_loads = (TILE_M * QK4_1) / (THREADS_M * THREADS_N);
        #pragma unroll 4
        for (int load_idx = 0; load_idx < act_loads; ++load_idx) {
            const int flat_idx = tid + load_idx * (THREADS_M * THREADS_N);
            const int m_local = flat_idx / QK4_1;
            const int k_local = flat_idx % QK4_1;
            const int m_global = block_m * TILE_M + m_local;
            const int k_global = k_block * QK4_1 + k_local;

            if (m_global < M && k_global < K) {
                smem_act[m_local][k_local] = activation[m_global * K + k_global];
            } else {
                smem_act[m_local][k_local] = 0.0f;
            }
        }

        __syncthreads();

        if (thread_m < items_per_thread_m) {
            for (int m_offset = 0; m_offset < TILE_M / items_per_thread_m; ++m_offset) {
                const int m_local = thread_m + m_offset * items_per_thread_m;
                if (m_local >= TILE_M) continue;

                float local_max = 0.0f;
                float local_sum = 0.0f;

                for (int k = 0; k < QK4_1; ++k) {
                    float val = smem_act[m_local][k];
                    local_max = fmaxf(local_max, fabsf(val));
                    local_sum += val;
                }

                if (thread_n == 0) {
                    float d_a = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
                    smem_act_scale[m_local] = d_a;
                    smem_act_sum[m_local] = local_sum;
                }
            }
        }

        __syncthreads();

        #pragma unroll 4
        for (int load_idx = 0; load_idx < act_loads; ++load_idx) {
            const int flat_idx = tid + load_idx * (THREADS_M * THREADS_N);
            const int m_local = flat_idx / QK4_1;
            const int k_local = flat_idx % QK4_1;

            if (m_local < TILE_M && k_local < QK4_1) {
                float val = smem_act[m_local][k_local];
                float d_a = smem_act_scale[m_local];
                smem_act_q[m_local][k_local] = (int8_t)__float2int_rn(val / d_a);
            }
        }

        for (int n_local = tid; n_local < TILE_N; n_local += THREADS_M * THREADS_N) {
            const int n_global = block_n * TILE_N + n_local;
            if (n_global < N) {
                smem_weight[n_local] = weight[n_global * num_k_blocks + k_block];
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < items_per_thread_m; ++i) {
            const int m_local = thread_m * items_per_thread_m + i;
            const int m_global = block_m * TILE_M + m_local;
            if (m_global >= M) continue;

            float d_a = smem_act_scale[m_local];
            float s_a = smem_act_sum[m_local];

            #pragma unroll
            for (int j = 0; j < items_per_thread_n; ++j) {
                const int n_local = thread_n * items_per_thread_n + j;
                const int n_global = block_n * TILE_N + n_local;
                if (n_global >= N) continue;

                const block_q4_1* w_block = &smem_weight[n_local];
                float d_w = half2float(w_block->d);
                float m_w = half2float(w_block->m);
                if (d_w == 0.0f) d_w = 1.0f;

                int32_t sumi = 0;

                #pragma unroll 4
                for (int ii = 0; ii < 4; ++ii) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_act_q[m_local][ii * 4]);
                    uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);
                    int w_pack = (int)(w_packed & 0x0F) | (((int)(w_packed >> 8) & 0x0F) << 8) |
                                 (((int)(w_packed >> 16) & 0x0F) << 16) | (((int)(w_packed >> 24) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll 4
                for (int ii = 0; ii < 4; ++ii) {
                    int a_pack = *reinterpret_cast<const int*>(&smem_act_q[m_local][16 + ii * 4]);
                    uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[ii * 4]);
                    int w_pack = (int)((w_packed >> 4) & 0x0F) | (((int)(w_packed >> 12) & 0x0F) << 8) |
                                 (((int)(w_packed >> 20) & 0x0F) << 16) | (((int)(w_packed >> 28) & 0x0F) << 24);
                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                accum[i][j] += d_w * d_a * (float)sumi + m_w * s_a;
            }
        }

        __syncthreads();
    }

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
// Host wrapper
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const int num_k_blocks = K / 32;

    torch::Tensor weight_reshaped;
    if (weight.dim() == 1) {
        weight_reshaped = weight.view({N, num_k_blocks, 20});
    } else if (weight.dim() == 3) {
        weight_reshaped = weight;
    } else {
        AT_ASSERTM(false, "Weight must be 1D or 3D tensor");
    }
    weight_reshaped = weight_reshaped.contiguous().view({-1});

    if (M == 1) {
        int threads = 256;
        int blocks = (M * N + threads - 1) / threads;
        w4a32c8_q4_1_kernel_simple<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight_reshaped.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks(
            (M + TILE_M - 1) / TILE_M,
            (N + TILE_N - 1) / TILE_N
        );

        w4a32c8_q4_1_kernel_tiled_opt<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight_reshaped.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    AT_ASSERTM(err == cudaSuccess, "CUDA kernel failed: " + std::string(cudaGetErrorString(err)));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_1 GEMM v3 - Optimized");
}
