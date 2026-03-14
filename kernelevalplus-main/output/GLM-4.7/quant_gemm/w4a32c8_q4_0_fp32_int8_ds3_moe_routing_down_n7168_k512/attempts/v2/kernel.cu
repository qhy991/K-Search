#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

// BLOCK_Q4_0 format: 18 bytes per block (llama.cpp compatible)
// Each block contains 32 quantized 4-bit values
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
// The lower nibble comes first in each byte
__device__ inline int8_t unpack_q4_0_llama(const uint8_t* qs, int idx) {
    if (idx < 16) {
        // First 16 values: take lower nibble of each byte
        return static_cast<int8_t>(static_cast<int32_t>(qs[idx] & 0x0F) - 8);
    } else {
        // Last 16 values: take upper nibble of each byte
        return static_cast<int8_t>(static_cast<int32_t>((qs[idx - 16] >> 4) & 0x0F) - 8);
    }
}

// Helper: pack 8-bit values into 32-bit for dp4a
__device__ inline int32_t pack8_to_32(int8_t a, int8_t b, int8_t c, int8_t d) {
    uint32_t pa = static_cast<uint32_t>(static_cast<uint8_t>(a));
    uint32_t pb = static_cast<uint32_t>(static_cast<uint8_t>(b));
    uint32_t pc = static_cast<uint32_t>(static_cast<uint8_t>(c));
    uint32_t pd = static_cast<uint32_t>(static_cast<uint8_t>(d));
    return static_cast<int32_t>((pd << 24) | (pc << 16) | (pb << 8) | pa);
}

// Compute dot product using dp4a instruction (4 pairs at a time)
__device__ inline int32_t dot_product_dp4a(const int8_t* a, const int8_t* b) {
    int32_t sum = 0;
    // Process 8 groups of 4 values
    sum = __dp4a(pack8_to_32(a[0], a[1], a[2], a[3]), pack8_to_32(b[0], b[1], b[2], b[3]), sum);
    sum = __dp4a(pack8_to_32(a[4], a[5], a[6], a[7]), pack8_to_32(b[4], b[5], b[6], b[7]), sum);
    sum = __dp4a(pack8_to_32(a[8], a[9], a[10], a[11]), pack8_to_32(b[8], b[9], b[10], b[11]), sum);
    sum = __dp4a(pack8_to_32(a[12], a[13], a[14], a[15]), pack8_to_32(b[12], b[13], b[14], b[15]), sum);
    sum = __dp4a(pack8_to_32(a[16], a[17], a[18], a[19]), pack8_to_32(b[16], b[17], b[18], b[19]), sum);
    sum = __dp4a(pack8_to_32(a[20], a[21], a[22], a[23]), pack8_to_32(b[20], b[21], b[22], b[23]), sum);
    sum = __dp4a(pack8_to_32(a[24], a[25], a[26], a[27]), pack8_to_32(b[24], b[25], b[26], b[27]), sum);
    sum = __dp4a(pack8_to_32(a[28], a[29], a[30], a[31]), pack8_to_32(b[28], b[29], b[30], b[31]), sum);
    return sum;
}

// Strategy 1: Simple kernel for very small batches (M <= 4)
// Optimized for memory bandwidth with one thread per output
__global__ void gemm_q4_0_fp32_small_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;

    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sum = 0.0f;

    // Iterate over K in blocks of 32
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Load 32 activation values
        float a_max = 0.0f;
        float a_vals[32];

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = activation[m * K + kb * 32 + i];
            a_vals[i] = a_val;
            a_max = fmaxf(a_max, fabsf(a_val));
        }

        // Q8_1-style activation quantization: find max, then scale
        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        // Load weight block
        const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
        float w_scale = read_half_as_float(w_block->d);

        // Unpack weight values once
        int8_t w_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            w_q[i] = unpack_q4_0_llama(w_block->qs, i);
        }

        // Compute dot product with quantized values
        int32_t dot_i = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int8_t a_q = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
            dot_i += static_cast<int32_t>(a_q) * static_cast<int32_t>(w_q[i]);
        }

        // Apply scales: result = a_scale * w_scale * dot_i
        sum += a_scale * w_scale * static_cast<float>(dot_i);
    }

    output[m * N + n] = sum;
}

// Strategy 2: Use dp4a for compute-bound medium batches (4 < M <= 32)
__global__ void gemm_q4_0_fp32_medium_m_dp4a(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int N_PER_THREAD = 4;

    int m = blockIdx.y;
    int n_base = blockIdx.x * blockDim.x * N_PER_THREAD + threadIdx.x * N_PER_THREAD;

    if (m >= M) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sums[N_PER_THREAD] = {0.0f};

    // Iterate over K in blocks of 32
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Load activation block once (shared by all N_PER_THREAD outputs)
        float a_max = 0.0f;
        float a_vals[32];

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = activation[m * K + kb * 32 + i];
            a_vals[i] = a_val;
            a_max = fmaxf(a_max, fabsf(a_val));
        }

        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        // Pre-quantize activations
        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
        }

        // Compute for N_PER_THREAD outputs using dp4a
        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
            int n = n_base + j;
            if (n >= N) continue;

            const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);

            // Unpack weight values
            int8_t w_q[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                w_q[i] = unpack_q4_0_llama(w_block->qs, i);
            }

            // Use dp4a for faster dot product
            int32_t dot_i = dot_product_dp4a(a_q, w_q);

            sums[j] += a_scale * w_scale * static_cast<float>(dot_i);
        }
    }

    // Write results
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) {
        int n = n_base + j;
        if (n < N) {
            output[m * N + n] = sums[j];
        }
    }
}

// Strategy 3: Shared memory tiled for larger batches (32 < M <= 128)
__global__ void gemm_q4_0_fp32_large_m_tiled(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int N_PER_THREAD = 8;
    const int TILE_N = 64;  // Process 64 outputs per block

    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n_block = blockIdx.x * TILE_N;
    int n_local = threadIdx.x * N_PER_THREAD;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    __shared__ float s_a_vals[32];  // Shared activation values

    if (m >= M) return;

    float sums[N_PER_THREAD] = {0.0f};

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Load activation values to shared memory (only first thread in y dimension loads)
        if (threadIdx.y == 0) {
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                s_a_vals[i] = a_val;
                a_max = fmaxf(a_max, fabsf(a_val));
            }
            // Store scale in the last element
            s_a_vals[0] = a_max;  // Use first element for max, will recalc scale
        }
        __syncthreads();

        // Each thread computes its own scale from shared values
        float a_max = s_a_vals[0];
        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        // Pre-quantize activations
        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(s_a_vals[i] / a_scale)));
        }

        // Compute for N_PER_THREAD outputs
        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
            int n = n_block + n_local + j;
            if (n >= N) continue;

            const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);

            // Unpack weight values
            int8_t w_q[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                w_q[i] = unpack_q4_0_llama(w_block->qs, i);
            }

            // Use dp4a for dot product
            int32_t dot_i = dot_product_dp4a(a_q, w_q);

            sums[j] += a_scale * w_scale * static_cast<float>(dot_i);
        }
    }

    // Write results
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) {
        int n = n_block + n_local + j;
        if (n < N) {
            output[m * N + n] = sums[j];
        }
    }
}

// Strategy 4: For very large batches (M > 128), use 2D tiling
__global__ void gemm_q4_0_fp32_very_large_m_2d(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int N_PER_THREAD = 4;
    const int M_PER_WARP = 4;

    int warp_id = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int lane_id = threadIdx.x;
    int m_base = warp_id * M_PER_WARP;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Shared memory for activation tile
    __shared__ float s_a[32][4];  // 32 values, 4 warps

    for (int m_offset = 0; m_offset < M_PER_WARP; ++m_offset) {
        int m = m_base + m_offset;
        if (m >= M) continue;

        int n_base = blockIdx.x * blockDim.x * N_PER_THREAD + lane_id * N_PER_THREAD;

        float sums[N_PER_THREAD] = {0.0f};

        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            // Load activation to shared memory
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                s_a[i][m_offset] = a_val;
                a_max = fmaxf(a_max, fabsf(a_val));
            }

            float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

            // Pre-quantize
            int8_t a_q[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(s_a[i][m_offset] / a_scale)));
            }

            #pragma unroll
            for (int j = 0; j < N_PER_THREAD; ++j) {
                int n = n_base + j;
                if (n >= N) continue;

                const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
                float w_scale = read_half_as_float(w_block->d);

                int8_t w_q[32];
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    w_q[i] = unpack_q4_0_llama(w_block->qs, i);
                }

                int32_t dot_i = dot_product_dp4a(a_q, w_q);
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
    torch::Tensor weight,     // (N, K/32) uint8, packed BLOCK_Q4_0
    torch::Tensor activation, // (M, K) float32
    int M, int N, int K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* w_ptr = static_cast<const uint8_t*>(weight.data_ptr<uint8_t>());
    const float* a_ptr = activation.data_ptr<float>();
    float* o_ptr = output.data_ptr<float>();

    // Choose strategy based on M
    if (M <= 4) {
        // Strategy 1: Small M - one thread per output
        const int THREADS_PER_BLOCK = 256;
        dim3 grid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, M);
        dim3 block(THREADS_PER_BLOCK);
        gemm_q4_0_fp32_small_m<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else if (M <= 32) {
        // Strategy 2: Medium M - use dp4a
        const int THREADS_PER_BLOCK = 128;
        const int N_PER_THREAD = 4;
        dim3 grid((N + THREADS_PER_BLOCK * N_PER_THREAD - 1) / (THREADS_PER_BLOCK * N_PER_THREAD), M);
        dim3 block(THREADS_PER_BLOCK);
        gemm_q4_0_fp32_medium_m_dp4a<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else if (M <= 128) {
        // Strategy 3: Large M - shared memory tiling
        const int BLOCK_X = 8;
        const int BLOCK_Y = 8;
        const int TILE_N = 64;
        const int N_PER_THREAD = 8;
        dim3 grid((N + TILE_N - 1) / TILE_N, M);
        dim3 block(BLOCK_X, BLOCK_Y);
        gemm_q4_0_fp32_large_m_tiled<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else {
        // Strategy 4: Very large M - 2D tiling
        const int BLOCK_X = 32;
        const int BLOCK_Y = 32;
        const int N_PER_THREAD = 4;
        dim3 grid((N + BLOCK_X * N_PER_THREAD - 1) / (BLOCK_X * N_PER_THREAD), (M + 3) / 4);
        dim3 block(BLOCK_X, BLOCK_Y);
        gemm_q4_0_fp32_very_large_m_2d<<<grid, block>>>(w_ptr, a_ptr, o_ptr, M, N, K);
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
