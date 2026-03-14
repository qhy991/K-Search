#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// BLOCK_Q4_0 format: 18 bytes per block
typedef struct {
    uint16_t d;      // scale (fp16)
    uint8_t qs[16];  // packed quanta (32 x 4-bit values)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size must be 18 bytes");

// Helper to read fp16 scale as float32
__device__ inline float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Helper: unpack 4-bit values with llama.cpp-compatible ordering
// llama.cpp ordering: first all low nibbles (indices 0-15), then all high nibbles (indices 16-31)
__device__ inline int8_t unpack_q4_0_llama(const uint8_t* qs, int idx) {
    if (idx < 16) {
        return static_cast<int8_t>(static_cast<int32_t>(qs[idx] & 0x0F) - 8);
    } else {
        return static_cast<int8_t>(static_cast<int32_t>((qs[idx - 16] >> 4) & 0x0F) - 8);
    }
}

// Optimized: Process 8 outputs per thread for better memory coalescing
__global__ void gemm_q4_0_fp32_simple_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int N_PER_THREAD = 4;  // Process 4 outputs per thread

    int m = blockIdx.y;
    int n_base = blockIdx.x * blockDim.x * N_PER_THREAD + threadIdx.x * N_PER_THREAD;

    if (m >= M) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Process N_PER_THREAD outputs
    float sums[N_PER_THREAD] = {0.0f};

    // Iterate over K blocks
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Load activation block once
        float a_block[32];
        float a_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = activation[m * K + kb * 32 + i];
            a_block[i] = a_val;
            a_max = fmaxf(a_max, fabsf(a_val));
        }

        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        // Pre-quantize activation values
        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_block[i] / a_scale)));
        }

        // Compute dot products for N_PER_THREAD outputs
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

    // Write results
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) {
        int n = n_base + j;
        if (n < N) {
            output[m * N + n] = sums[j];
        }
    }
}

// Optimized tiled kernel with better memory access patterns
__global__ void gemm_q4_0_fp32_tiled_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;

    // Tile dimensions - adjusted for better occupancy
    const int TILE_M = 8;
    const int TILE_N = 16;

    // Shared memory
    __shared__ float s_activation[TILE_M * K_BLOCK];
    __shared__ float s_weight_scale[TILE_N];
    __shared__ int8_t s_weight_q[TILE_N * K_BLOCK];

    int tile_m = blockIdx.y * TILE_M;
    int tile_n = blockIdx.x * TILE_N;

    int lane_m = threadIdx.y;
    int lane_n = threadIdx.x;

    int m = tile_m + lane_m;
    int n = tile_n + lane_n;

    // Early exit for out-of-bounds threads
    bool valid = (m < M && n < N);
    if (!valid) {
        if (m < M && n < N) output[m * N + n] = 0.0f;
        return;
    }

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sum = 0.0f;

    // Iterate over K blocks
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Cooperative load of activation tile with vectorized reads
        if (lane_n < 2) {  // Each thread loads 2 values
            for (int i = 0; i < 16; ++i) {
                int load_idx = lane_m * K_BLOCK + lane_n * 16 + i;
                int a_idx = (tile_m + lane_m) * K + kb * K_BLOCK + lane_n * 16 + i;
                if ((tile_m + lane_m) < M && (lane_n * 16 + i) < K_BLOCK) {
                    s_activation[load_idx] = activation[a_idx];
                }
            }
        }

        // Load weight scales - coalesced access
        if (lane_n < TILE_N && tile_n + lane_n < N) {
            const block_q4_0* w_block = &w_blocks[(tile_n + lane_n) * K_BLOCKS + kb];
            s_weight_scale[lane_n] = read_half_as_float(w_block->d);
        }

        // Load and unpack weight quanta
        if (lane_n < TILE_N && tile_n + lane_n < N) {
            const block_q4_0* w_block = &w_blocks[(tile_n + lane_n) * K_BLOCKS + kb];
            // Unpack with llama.cpp ordering - all low nibbles first, then high nibbles
            int8_t* w_q_ptr = &s_weight_q[lane_n * K_BLOCK];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                w_q_ptr[i] = static_cast<int8_t>(static_cast<int32_t>(w_block->qs[i] & 0x0F) - 8);
                w_q_ptr[i + 16] = static_cast<int8_t>(static_cast<int32_t>((w_block->qs[i] >> 4) & 0x0F) - 8);
            }
        }

        __syncthreads();

        // Compute activation scale once per row
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < K_BLOCK; ++i) {
            a_max = fmaxf(a_max, fabsf(s_activation[lane_m * K_BLOCK + i]));
        }
        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        // Dot product with loop unrolling
        int32_t dot_i = 0;
        const float* a_ptr = &s_activation[lane_m * K_BLOCK];
        const int8_t* w_ptr = &s_weight_q[lane_n * K_BLOCK];

        #pragma unroll 8
        for (int i = 0; i < K_BLOCK; ++i) {
            int8_t a_q = static_cast<int8_t>(roundf(__float2int_rn(a_ptr[i] / a_scale)));
            dot_i += static_cast<int32_t>(a_q) * static_cast<int32_t>(w_ptr[i]);
        }

        sum += a_scale * s_weight_scale[lane_n] * static_cast<float>(dot_i);

        __syncthreads();
    }

    output[m * N + n] = sum;
}

// Highly optimized kernel for large batches using vectorized loads
__global__ void gemm_q4_0_fp32_large_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;

    // Process multiple M rows per thread block
    const int M_PER_BLOCK = 16;
    const int N_PER_BLOCK = 8;

    int m_base = blockIdx.y * M_PER_BLOCK;
    int n_base = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_base >= M || n_base >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Process M_PER_BLOCK rows
    for (int m_rel = 0; m_rel < M_PER_BLOCK; ++m_rel) {
        int m = m_base + m_rel;
        if (m >= M) break;

        int n = n_base;
        if (n >= N) continue;

        float sum = 0.0f;

        // Process all K blocks
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            // Compute activation scale
            float a_max = 0.0f;
            float a_vals[32];

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                a_vals[i] = a_val;
                a_max = fmaxf(a_max, fabsf(a_val));
            }

            float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

            // Load weight and compute dot product
            const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);

            int32_t dot_i = 0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int8_t a_q = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
                int8_t w_q = unpack_q4_0_llama(w_block->qs, i);
                dot_i += static_cast<int32_t>(a_q) * static_cast<int32_t>(w_q);
            }

            sum += a_scale * w_scale * static_cast<float>(dot_i);
        }

        output[m * N + n] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // Strategy dispatch with optimized thresholds
    if (M <= 8) {
        // Very small batch - optimized for bandwidth
        int threads_per_block = 256;
        int n_per_thread = 4;
        dim3 block(threads_per_block / n_per_thread);
        dim3 grid((N + n_per_thread * block.x - 1) / (n_per_thread * block.x), M);

        gemm_q4_0_fp32_simple_v2<<<grid, block>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    } else if (M <= 64) {
        // Medium batch - tiled approach
        dim3 block(16, 8);  // TILE_N x TILE_M
        dim3 grid((N + 16 - 1) / 16, (M + 8 - 1) / 8);

        gemm_q4_0_fp32_tiled_v2<<<grid, block>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    } else {
        // Large batch - optimized for compute throughput
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x, (M + 16 - 1) / 16);

        gemm_q4_0_fp32_large_batch<<<grid, block>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
