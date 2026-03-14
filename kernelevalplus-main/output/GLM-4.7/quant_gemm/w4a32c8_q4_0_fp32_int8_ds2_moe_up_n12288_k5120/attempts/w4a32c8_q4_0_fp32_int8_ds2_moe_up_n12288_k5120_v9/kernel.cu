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

// Unpack 4 Q4_0 values into 4 int8 values for dp4a
__device__ inline void unpack_4_q4_0(const uint8_t* qs, int base_idx, int8_t out[4]) {
    // base_idx 0-15: low nibbles; base_idx 16-31: high nibbles
    // For dp4a, we want 4 consecutive values
    for (int i = 0; i < 4; ++i) {
        int idx = base_idx + i;
        if (idx < 16) {
            out[i] = static_cast<int8_t>(static_cast<int32_t>(qs[idx] & 0x0F) - 8);
        } else {
            out[i] = static_cast<int8_t>(static_cast<int32_t>((qs[idx - 16] >> 4) & 0x0F) - 8);
        }
    }
}

// Optimized kernel using dp4a for vectorized dot product
__global__ void gemm_q4_0_fp32_dp4a(
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

    const block_q4_0* __restrict__ w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sum = 0.0f;

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        const float* __restrict__ a_ptr = &activation[m * K + kb * K_BLOCK];
        float a_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_ptr[i]));
        }

        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        // Quantize activation to int8
        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_ptr[i] / a_scale)));
        }

        const block_q4_0* __restrict__ w_block = &w_blocks[n * K_BLOCKS + kb];
        float w_scale = read_half_as_float(w_block->d);

        // Use dp4a for dot product (8 dp4a instructions for 32 elements)
        int32_t dot_i = 0;
        const uint8_t* w_qs = w_block->qs;

        // Process low nibbles (indices 0-15) using 4 dp4a calls
        int32_t w_low[4];  // Each int32 holds 4 unpacked low nibbles
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int32_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int8_t val = static_cast<int8_t>(static_cast<int32_t>(w_qs[i * 4 + j] & 0x0F) - 8);
                packed |= (static_cast<int32_t>(val) & 0xFF) << (j * 8);
            }
            w_low[i] = packed;
        }

        // Process high nibbles (indices 16-31) using 4 dp4a calls
        int32_t w_high[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int32_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int8_t val = static_cast<int8_t>(static_cast<int32_t>((w_qs[i * 4 + j] >> 4) & 0x0F) - 8);
                packed |= (static_cast<int32_t>(val) & 0xFF) << (j * 8);
            }
            w_high[i] = packed;
        }

        // Use dp4a for computation: 8 calls total (4 for low, 4 for high)
        int32_t a_low[4], a_high[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a_low[i] = 0;
            a_high[i] = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                a_low[i] |= (static_cast<int32_t>(a_q[i * 4 + j]) & 0xFF) << (j * 8);
                a_high[i] |= (static_cast<int32_t>(a_q[16 + i * 4 + j]) & 0xFF) << (j * 8);
            }
        }

        // dp4a: dot product accumulate 4 pairs of int8 values
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dot_i += __dp4a(a_low[i], w_low[i], 0);
            dot_i += __dp4a(a_high[i], w_high[i], 0);
        }

        sum += a_scale * w_scale * static_cast<float>(dot_i);
    }

    output[m * N + n] = sum;
}

// Strategy 2: Multi-output per thread using dp4a
__global__ void gemm_q4_0_fp32_dp4a_multi(
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

    const block_q4_0* __restrict__ w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sums[N_PER_THREAD] = {0.0f};

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        const float* __restrict__ a_ptr = &activation[m * K + kb * K_BLOCK];
        float a_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_ptr[i]));
        }

        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);

        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = static_cast<int8_t>(roundf(__float2int_rn(a_ptr[i] / a_scale)));
        }

        // Pack activation values for dp4a
        int32_t a_low[4], a_high[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a_low[i] = 0;
            a_high[i] = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                a_low[i] |= (static_cast<int32_t>(a_q[i * 4 + j]) & 0xFF) << (j * 8);
                a_high[i] |= (static_cast<int32_t>(a_q[16 + i * 4 + j]) & 0xFF) << (j * 8);
            }
        }

        #pragma unroll
        for (int j = 0; j < N_PER_THREAD; ++j) {
            int n = n_base + j;
            if (n >= N) continue;

            const block_q4_0* __restrict__ w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);
            const uint8_t* w_qs = w_block->qs;

            // Pack weight values for dp4a
            int32_t w_low[4], w_high[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                w_low[i] = 0;
                w_high[i] = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int8_t low_val = static_cast<int8_t>(static_cast<int32_t>(w_qs[i * 4 + j] & 0x0F) - 8);
                    int8_t high_val = static_cast<int8_t>(static_cast<int32_t>((w_qs[i * 4 + j] >> 4) & 0x0F) - 8);
                    w_low[i] |= (static_cast<int32_t>(low_val) & 0xFF) << (j * 8);
                    w_high[i] |= (static_cast<int32_t>(high_val) & 0xFF) << (j * 8);
                }
            }

            int32_t dot_i = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                dot_i += __dp4a(a_low[i], w_low[i], 0);
                dot_i += __dp4a(a_high[i], w_high[i], 0);
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
        // Use dp4a single-output for small batches
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x, M);
        gemm_q4_0_fp32_dp4a<<<grid, block>>>(w_ptr, act_ptr, out_ptr, M, N, K);
    } else {
        // Use dp4a multi-output for larger batches
        int n_per_thread = 4;
        dim3 block(256 / n_per_thread);
        dim3 grid((N + n_per_thread * block.x - 1) / (n_per_thread * block.x), M);
        gemm_q4_0_fp32_dp4a_multi<<<grid, block>>>(w_ptr, act_ptr, out_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM - DP4A Optimized");
}
