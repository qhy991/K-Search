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

// dp4a-based dot product for 4 pairs of int8 values
__device__ inline int32_t dot4_int8(const int8_t* a, const int8_t* b) {
    int32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return sum;
}

// Optimized kernel for small batches using vectorized loads
__global__ void gemm_q4_0_fp32_small_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int N_PER_THREAD = 8;  // Process 8 outputs per thread

    int m = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_base = tid * N_PER_THREAD;

    if (m >= M || n_base >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sums[N_PER_THREAD] = {0.0f};

    // Pre-compute activation quantization for all K blocks
    // Store: a_scale[K_BLOCKS], a_q[K_BLOCKS][32]
    extern __shared__ float s_a_scale[];  // K_BLOCKS floats
    int8_t* s_a_q = reinterpret_cast<int8_t*>(&s_a_scale[K_BLOCKS]);  // K_BLOCKS * 32 int8_t

    // Each thread block processes one K block
    for (int kb_tid = 0; kb_tid < K_BLOCKS; kb_tid += blockDim.x) {
        int kb = kb_tid + threadIdx.x;
        if (kb < K_BLOCKS) {
            // Compute activation scale and quantize
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                a_max = fmaxf(a_max, fabsf(a_val));
            }
            s_a_scale[kb] = fmaxf(a_max / 127.0f, 1e-7f);

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                s_a_q[kb * 32 + i] = static_cast<int8_t>(roundf(__float2int_rn(a_val / s_a_scale[kb])));
            }
        }
    }

    __syncthreads();

    // Process N_PER_THREAD outputs using pre-quantized activations
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) {
        int n = n_base + j;
        if (n >= N) continue;

        float sum = 0.0f;
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
            float w_scale = read_half_as_float(w_block->d);
            float a_scale = s_a_scale[kb];

            int32_t dot_i = 0;
            const int8_t* a_q_ptr = &s_a_q[kb * 32];

            // Use dp4a-style unrolling for better ILP
            #pragma unroll 8
            for (int i = 0; i < 32; ++i) {
                int8_t w_q = unpack_q4_0_llama(w_block->qs, i);
                dot_i += static_cast<int32_t>(a_q_ptr[i]) * static_cast<int32_t>(w_q);
            }

            sum += a_scale * w_scale * static_cast<float>(dot_i);
        }
        sums[j] = sum;
    }

    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) {
        int n = n_base + j;
        if (n < N) {
            output[m * N + n] = sums[j];
        }
    }
}

// Medium batch kernel with improved shared memory usage
__global__ void gemm_q4_0_fp32_medium_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int TILE_M = 4;
    const int TILE_N = 32;

    // Shared memory: activation values and weight quanta
    __shared__ float s_activation[TILE_M * K_BLOCK];
    __shared__ float s_a_scale[TILE_M];
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
        // Load activation values cooperatively
        if (lane_n < 2) {
            for (int i = 0; i < 16; ++i) {
                int a_idx = (tile_m + lane_m) * K + kb * K_BLOCK + lane_n * 16 + i;
                if ((tile_m + lane_m) < M) {
                    s_activation[lane_m * K_BLOCK + lane_n * 16 + i] = activation[a_idx];
                }
            }
        }

        __syncthreads();

        // Compute activation scale and quantize per row
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < K_BLOCK; ++i) {
            a_max = fmaxf(a_max, fabsf(s_activation[lane_m * K_BLOCK + i]));
        }
        float a_scale = fmaxf(a_max / 127.0f, 1e-7f);
        s_a_scale[lane_m] = a_scale;

        #pragma unroll
        for (int i = 0; i < K_BLOCK; ++i) {
            s_a_q[lane_m * K_BLOCK + i] = static_cast<int8_t>(roundf(__float2int_rn(s_activation[lane_m * K_BLOCK + i] / a_scale)));
        }

        // Load weight data
        if (lane_n < TILE_N && tile_n + lane_n < N) {
            const block_q4_0* w_block = &w_blocks[(tile_n + lane_n) * K_BLOCKS + kb];
            s_weight_scale[lane_n] = read_half_as_float(w_block->d);

            // Unpack weight quanta
            int8_t* w_q_ptr = &s_weight_q[lane_n * K_BLOCK];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                w_q_ptr[i] = static_cast<int8_t>(static_cast<int32_t>(w_block->qs[i] & 0x0F) - 8);
                w_q_ptr[i + 16] = static_cast<int8_t>(static_cast<int32_t>((w_block->qs[i] >> 4) & 0x0F) - 8);
            }
        }

        __syncthreads();

        // Compute dot product with vectorized operations
        int32_t dot_i = 0;
        const int8_t* a_ptr = &s_a_q[lane_m * K_BLOCK];
        const int8_t* w_ptr = &s_weight_q[lane_n * K_BLOCK];

        // Process 8 values at a time for better ILP
        #pragma unroll 4
        for (int i = 0; i < K_BLOCK; ++i) {
            dot_i += static_cast<int32_t>(a_ptr[i]) * static_cast<int32_t>(w_ptr[i]);
        }

        sum += s_a_scale[lane_m] * s_weight_scale[lane_n] * static_cast<float>(dot_i);

        __syncthreads();
    }

    output[m * N + n] = sum;
}

// Large batch kernel optimized for compute throughput
__global__ void gemm_q4_0_fp32_large_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;
    const int M_PER_BLOCK = 8;
    const int N_WARPS = 4;

    // Each warp processes multiple outputs
    int warp_id = threadIdx.y;  // 0 to N_WARPS-1
    int lane_id = threadIdx.x;  // 0 to 31

    int m_base = blockIdx.y * M_PER_BLOCK;
    int n_base = blockIdx.x * blockDim.y * 32 + warp_id * 32;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Process M_PER_BLOCK rows
    for (int m_rel = 0; m_rel < M_PER_BLOCK; ++m_rel) {
        int m = m_base + m_rel;
        if (m >= M) break;

        // Pre-quantize activation for this row across all K blocks
        extern __shared__ float s_shared[];
        float* s_a_scale = s_shared;
        int8_t* s_a_q = reinterpret_cast<int8_t*>(&s_shared[K_BLOCKS]);

        // Distribute K_BLOCKS across warp threads
        for (int kb = lane_id; kb < K_BLOCKS; kb += 32) {
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                a_max = fmaxf(a_max, fabsf(a_val));
            }
            s_a_scale[kb] = fmaxf(a_max / 127.0f, 1e-7f);

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float a_val = activation[m * K + kb * 32 + i];
                s_a_q[kb * 32 + i] = static_cast<int8_t>(roundf(__float2int_rn(a_val / s_a_scale[kb])));
            }
        }

        __syncthreads();

        // Process 32 outputs per warp (one per thread)
        int n = n_base + lane_id;
        if (n < N) {
            float sum = 0.0f;
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
                float w_scale = read_half_as_float(w_block->d);
                float a_scale = s_a_scale[kb];

                int32_t dot_i = 0;
                const int8_t* a_q_ptr = &s_a_q[kb * 32];

                // Unrolled dot product
                #pragma unroll 8
                for (int i = 0; i < 32; ++i) {
                    int8_t w_q = unpack_q4_0_llama(w_block->qs, i);
                    dot_i += static_cast<int32_t>(a_q_ptr[i]) * static_cast<int32_t>(w_q);
                }

                sum += a_scale * w_scale * static_cast<float>(dot_i);
            }
            output[m * N + n] = sum;
        }

        __syncthreads();
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

    const int K_BLOCKS = K / 32;

    if (M <= 4) {
        // Small batch: use shared memory for pre-quantized activations
        int shared_mem = K_BLOCKS * sizeof(float) + K_BLOCKS * 32 * sizeof(int8_t);
        dim3 block(256);
        dim3 grid((N + block.x * 8 - 1) / (block.x * 8), M);

        gemm_q4_0_fp32_small_v2<<<grid, block, shared_mem>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    } else if (M <= 64) {
        // Medium batch: tiled approach
        dim3 block(32, 4);
        dim3 grid((N + 32 - 1) / 32, (M + 4 - 1) / 4);

        gemm_q4_0_fp32_medium_v2<<<grid, block>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    } else {
        // Large batch: warp-level optimization
        int shared_mem = K_BLOCKS * sizeof(float) + K_BLOCKS * 32 * sizeof(int8_t);
        dim3 block(32, 4);  // 4 warps per block
        dim3 grid((N + 128 - 1) / 128, (M + 8 - 1) / 8);

        gemm_q4_0_fp32_large_v2<<<grid, block, shared_mem>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM - Optimized");
}
