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
__device__ inline int8_t unpack_q4_0_llama(const uint8_t* qs, int idx) {
    if (idx < 16) {
        return static_cast<int8_t>(static_cast<int32_t>(qs[idx] & 0x0F) - 8);
    } else {
        return static_cast<int8_t>(static_cast<int32_t>((qs[idx - 16] >> 4) & 0x0F) - 8);
    }
}

// Optimized for very small batches - improved memory access pattern
__global__ void gemm_q4_0_fp32_small_opt(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int N_PER_THREAD = 4;

    int m = blockIdx.y;
    int tid = threadIdx.x;
    int n_base = (blockIdx.x * blockDim.x + tid) * N_PER_THREAD;

    if (m >= M || n_base >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sums[N_PER_THREAD] = {0.0f};

    // Process all K blocks
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Load and quantize activation block once
        float a_vals[32];
        float a_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = activation[m * K + kb * 32 + i];
            a_vals[i] = a_val;
            a_max = fmaxf(a_max, fabsf(a_val));
        }

        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        // Pre-quantize activation values
        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
        }

        // Compute dot products for N_PER_THREAD outputs
        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
            int n = n_base + j;
            if (n >= N) continue;

            const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);

            // Unrolled dot product for better ILP
            int32_t dot_i = 0;
            #pragma unroll 8
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

// Optimized tiled kernel with better shared memory usage
__global__ void gemm_q4_0_fp32_tiled_opt(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int TILE_M = 8;
    const int TILE_N = 32;

    __shared__ float s_activation[TILE_M * K_BLOCK];
    __shared__ int8_t s_a_q[TILE_M * K_BLOCK];
    __shared__ float s_weight_scale[TILE_N];
    __shared__ int8_t s_weight_q[TILE_N * K_BLOCK];

    int tile_m = blockIdx.y * TILE_M;
    int tile_n = blockIdx.x * TILE_N;

    int lane_m = threadIdx.y;
    int lane_n = threadIdx.x;

    int m = tile_m + lane_m;
    int n = tile_n + lane_n;

    if (m >= M || n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sum = 0.0f;

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Cooperative load of activation tile
        if (lane_n < 2) {
            int4* act_vec = reinterpret_cast<int4*>(&s_activation[lane_m * K_BLOCK]);
            const int4* src_vec = reinterpret_cast<const int4*>(&activation[(tile_m + lane_m) * K + kb * K_BLOCK + lane_n * 16]);
            if ((tile_m + lane_m) < M) {
                act_vec[lane_n] = src_vec[0];
            }
        }

        __syncthreads();

        // Compute activation scale and quantize (once per row)
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < K_BLOCK; ++i) {
            a_max = fmaxf(a_max, fabsf(s_activation[lane_m * K_BLOCK + i]));
        }
        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        #pragma unroll
        for (int i = 0; i < K_BLOCK; ++i) {
            s_a_q[lane_m * K_BLOCK + i] = static_cast<int8_t>(roundf(__float2int_rn(s_activation[lane_m * K_BLOCK + i] / a_scale)));
        }

        // Load weight scales
        if (lane_n < TILE_N && tile_n + lane_n < N) {
            const block_q4_0* w_block = &w_blocks[(tile_n + lane_n) * K_BLOCKS + kb];
            s_weight_scale[lane_n] = read_half_as_float(w_block->d);
        }

        // Load and unpack weight quanta
        if (lane_n < TILE_N && tile_n + lane_n < N) {
            const block_q4_0* w_block = &w_blocks[(tile_n + lane_n) * K_BLOCKS + kb];
            // Vectorized load of 16 bytes
            const int4* w_qs_vec = reinterpret_cast<const int4*>(w_block->qs);
            int4* dst_vec = reinterpret_cast<int4*>(&s_weight_q[lane_n * K_BLOCK]);
            dst_vec[0] = w_qs_vec[0];  // Load 16 bytes
            // Unpack in-place
            int8_t* w_q_ptr = &s_weight_q[lane_n * K_BLOCK];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int8_t low = w_q_ptr[i] & 0x0F;
                int8_t high = (w_q_ptr[i] >> 4) & 0x0F;
                w_q_ptr[i] = static_cast<int8_t>(static_cast<int32_t>(low) - 8);
                w_q_ptr[i + 16] = static_cast<int8_t>(static_cast<int32_t>(high) - 8);
            }
        }

        __syncthreads();

        // Compute dot product
        int32_t dot_i = 0;
        const int8_t* a_ptr = &s_a_q[lane_m * K_BLOCK];
        const int8_t* w_ptr = &s_weight_q[lane_n * K_BLOCK];

        #pragma unroll 8
        for (int i = 0; i < K_BLOCK; ++i) {
            dot_i += static_cast<int32_t>(a_ptr[i]) * static_cast<int32_t>(w_ptr[i]);
        }

        sum += a_scale * s_weight_scale[lane_n] * static_cast<float>(dot_i);

        __syncthreads();
    }

    output[m * N + n] = sum;
}

// Optimized for large batches - better compute efficiency
__global__ void gemm_q4_0_fp32_large_opt(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int M_PER_BLOCK = 16;
    const int N_PER_THREAD = 1;

    int m_base = blockIdx.y * M_PER_BLOCK;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_base >= M || n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Process M_PER_BLOCK rows to amortize weight loading
    for (int m_rel = 0; m_rel < M_PER_BLOCK; ++m_rel) {
        int m = m_base + m_rel;
        if (m >= M) break;

        float sum = 0.0f;

        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            // Load activation values
            float a_max = 0.0f;
            float a_vals[32];

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                a_vals[i] = a_val;
                a_max = fmaxf(a_max, fabsf(a_val));
            }

            float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

            const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);

            // Pre-unpack weight quanta once
            int8_t w_q[32];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                w_q[i] = static_cast<int8_t>(static_cast<int32_t>(w_block->qs[i] & 0x0F) - 8);
                w_q[i + 16] = static_cast<int8_t>(static_cast<int32_t>((w_block->qs[i] >> 4) & 0x0F) - 8);
            }

            // Unrolled dot product
            int32_t dot_i = 0;
            #pragma unroll 8
            for (int i = 0; i < 32; ++i) {
                int8_t a_q = static_cast<int8_t>(roundf(__float2int_rn(a_vals[i] / a_scale)));
                dot_i += static_cast<int32_t>(a_q) * static_cast<int32_t>(w_q[i]);
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

    if (M <= 8) {
        // Small batches: optimized with pre-quantization
        int n_per_thread = 4;
        dim3 block(256 / n_per_thread);
        dim3 grid((N + n_per_thread * block.x - 1) / (n_per_thread * block.x), M);
        gemm_q4_0_fp32_small_opt<<<grid, block>>>(w_ptr, act_ptr, out_ptr, M, N, K);
    } else if (M <= 128) {
        // Medium batches: tiled with shared memory
        dim3 block(32, 8);
        dim3 grid((N + 32 - 1) / 32, (M + 8 - 1) / 8);
        gemm_q4_0_fp32_tiled_opt<<<grid, block>>>(w_ptr, act_ptr, out_ptr, M, N, K);
    } else {
        // Large batches: compute-optimized
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x, (M + 16 - 1) / 16);
        gemm_q4_0_fp32_large_opt<<<grid, block>>>(w_ptr, act_ptr, out_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM - Optimized V2");
}
