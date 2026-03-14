#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

// BLOCK_Q4_0 format: 18 bytes per block (llama.cpp compatible)
typedef struct {
    uint16_t d;      // delta/scale (fp16)
    uint8_t qs[16];  // packed quanta (32 x 4-bit values, 2 per byte)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size must be 18 bytes");

// Helper to read fp16 scale as float32
__device__ inline float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Helper: unpack 4-bit values with llama.cpp-compatible ordering
__device__ inline int8_t unpack_q4_0_llama(const uint8_t* qs, int idx) {
    if (idx < 16) {
        return static_cast<int8_t>(static_cast<int32_t>(qs[idx] & 0x0F) - 8);
    } else {
        return static_cast<int8_t>(static_cast<int32_t>((qs[idx - 16] >> 4) & 0x0F) - 8);
    }
}

// Strategy 1: Very small batches (M <= 4) - FULLY UNROLLED for max performance
__global__ void gemm_q4_0_fp32_small_m_unrolled(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const float* a_ptr = activation + m * K;
    const block_q4_0* w_block = &w_blocks[n * 16];  // K_BLOCKS = 16

    float sum = 0.0f;

    // Completely unrolled K loop for K=512 (16 blocks of 32)
    for (int kb = 0; kb < 16; ++kb) {
        float a_max = 0.0f;
        float a_vals[32];

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = a_ptr[kb * 32 + i];
            a_vals[i] = a_val;
            a_max = fmaxf(a_max, fabsf(a_val));
        }

        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);
        float w_scale = read_half_as_float(w_block[kb].d);

        int32_t dot_i = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int8_t a_q = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
            int8_t w_q = unpack_q4_0_llama(w_block[kb].qs, i);
            dot_i += static_cast<int32_t>(a_q) * static_cast<int32_t>(w_q);
        }

        sum += a_scale * w_scale * static_cast<float>(dot_i);
    }

    output[m * N + n] = sum;
}

// Strategy 2: Medium batches (4 < M <= 32) - Loop-based for better register usage
__global__ void gemm_q4_0_fp32_medium_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = 16;
    const int N_PER_THREAD = 4;

    int m = blockIdx.y;
    int n_base = blockIdx.x * blockDim.x * N_PER_THREAD + threadIdx.x * N_PER_THREAD;

    if (m >= M) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);
    const float* a_ptr = activation + m * K;

    float sums[N_PER_THREAD] = {0.0f};

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        float a_max = 0.0f;
        float a_vals[32];

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = a_ptr[kb * 32 + i];
            a_vals[i] = a_val;
            a_max = fmaxf(a_max, fabsf(a_val));
        }

        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
        }

        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
            int n = n_base + j;
            if (n >= N) continue;

            const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);

            int32_t dot_i = 0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int8_t w_q = unpack_q4_0_llama(w_block->qs, i);
                dot_i += static_cast<int32_t>(a_q[i]) * static_cast<int32_t>(w_q);
            }

            sums[j] += a_scale * w_scale * static_cast<float>(dot_i);
        }
    }

    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) {
        int n = n_base + j;
        if (n < N) {
            output[m * N + n] = sums[j];
        }
    }
}

// Strategy 3: Large batches (32 < M <= 128) - Loop-based for compute-bound
__global__ void gemm_q4_0_fp32_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = 16;
    const int N_PER_THREAD = 8;
    const int M_PER_THREAD = 2;

    int m_base = (blockIdx.y * blockDim.y + threadIdx.y) * M_PER_THREAD;
    int n_base = blockIdx.x * blockDim.x * N_PER_THREAD + threadIdx.x * N_PER_THREAD;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    for (int m_offset = 0; m_offset < M_PER_THREAD; ++m_offset) {
        int m = m_base + m_offset;
        if (m >= M) continue;

        const float* a_ptr = activation + m * K;

        float sums[N_PER_THREAD] = {0.0f};

        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            float a_max = 0.0f;
            float a_vals[32];

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = a_ptr[kb * 32 + i];
                a_vals[i] = a_val;
                a_max = fmaxf(a_max, fabsf(a_val));
            }

            float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

            int8_t a_q[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
            }

            #pragma unroll
            for (int j = 0; j < N_PER_THREAD; ++j) {
                int n = n_base + j;
                if (n >= N) continue;

                const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
                float w_scale = read_half_as_float(w_block->d);

                int32_t dot_i = 0;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    int8_t w_q = unpack_q4_0_llama(w_block->qs, i);
                    dot_i += static_cast<int32_t>(a_q[i]) * static_cast<int32_t>(w_q);
                }

                sums[j] += a_scale * w_scale * static_cast<float>(dot_i);
            }
        }

        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
            int n = n_base + j;
            if (n < N) {
                output[m * N + n] = sums[j];
            }
        }
    }
}

// Strategy 4: Very large batches (M > 128) - Maximize throughput
__global__ void gemm_q4_0_fp32_very_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = 16;
    const int N_PER_THREAD = 4;
    const int M_PER_THREAD = 4;

    int m_base = (blockIdx.y * blockDim.y + threadIdx.y) * M_PER_THREAD;
    int n_base = blockIdx.x * blockDim.x * N_PER_THREAD + threadIdx.x * N_PER_THREAD;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    for (int m_offset = 0; m_offset < M_PER_THREAD; ++m_offset) {
        int m = m_base + m_offset;
        if (m >= M) continue;

        const float* a_ptr = activation + m * K;

        float sums[N_PER_THREAD] = {0.0f};

        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            float a_max = 0.0f;
            float a_vals[32];

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = a_ptr[kb * 32 + i];
                a_vals[i] = a_val;
                a_max = fmaxf(a_max, fabsf(a_val));
            }

            float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

            int8_t a_q[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
            }

            #pragma unroll
            for (int j = 0; j < N_PER_THREAD; ++j) {
                int n = n_base + j;
                if (n >= N) continue;

                const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
                float w_scale = read_half_as_float(w_block->d);

                int32_t dot_i = 0;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    int8_t w_q = unpack_q4_0_llama(w_block->qs, i);
                    dot_i += static_cast<int32_t>(a_q[i]) * static_cast<int32_t>(w_q);
                }

                sums[j] += a_scale * w_scale * static_cast<float>(dot_i);
            }
        }

        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
            int n = n_base + j;
            if (n < N) {
                output[m * N + n] = sums[j];
            }
        }
    }
}

// Host function to dispatch to appropriate kernel
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* w_ptr = static_cast<const uint8_t*>(weight.data_ptr<uint8_t>());
    const float* a_ptr = activation.data_ptr<float>();
    float* o_ptr = output.data_ptr<float>();

    if (M <= 4) {
        // Use fully unrolled kernel for memory-bound case
        const int THREADS_PER_BLOCK = 256;
        dim3 grid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, M);
        dim3 block(THREADS_PER_BLOCK);
        gemm_q4_0_fp32_small_m_unrolled<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else if (M <= 32) {
        const int THREADS_PER_BLOCK = 128;
        const int N_PER_THREAD = 4;
        dim3 grid((N + THREADS_PER_BLOCK * N_PER_THREAD - 1) / (THREADS_PER_BLOCK * N_PER_THREAD), M);
        dim3 block(THREADS_PER_BLOCK);
        gemm_q4_0_fp32_medium_m<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else if (M <= 128) {
        const int BLOCK_X = 16;
        const int BLOCK_Y = 8;
        const int N_PER_THREAD = 8;
        const int M_PER_THREAD = 2;
        dim3 grid((N + BLOCK_X * N_PER_THREAD - 1) / (BLOCK_X * N_PER_THREAD), (M + M_PER_THREAD - 1) / M_PER_THREAD);
        dim3 block(BLOCK_X, BLOCK_Y);
        gemm_q4_0_fp32_large_m<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else {
        const int BLOCK_X = 32;
        const int BLOCK_Y = 8;
        const int N_PER_THREAD = 4;
        const int M_PER_THREAD = 4;
        dim3 grid((N + BLOCK_X * N_PER_THREAD - 1) / (BLOCK_X * N_PER_THREAD), (M + M_PER_THREAD - 1) / M_PER_THREAD);
        dim3 block(BLOCK_X, BLOCK_Y);
        gemm_q4_0_fp32_very_large_m<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM W4A32C8");
}
