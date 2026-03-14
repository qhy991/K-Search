#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure (llama.cpp compatible)
// 18 bytes per block: 2 bytes FP16 scale + 16 bytes packed 4-bit values
struct block_q4_0 {
    uint16_t d;      // FP16 scale
    uint8_t qs[16];  // 32 packed quaternions
};

// Convert FP16 to FP32 using union for safety
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Optimized kernel for small M (1-8)
// Uses vectorized loads and larger thread blocks
__global__ void q4_0_gemm_small_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Each thread block processes 8 output columns
    // Each thread processes one output element
    const int tid = threadIdx.x;
    const int n = blockIdx.x * 8 + (tid % 8);
    const int m = blockIdx.y * 4 + (tid / 8);

    if (m >= M || n >= N) return;

    // Accumulator
    float sum = 0.0f;

    const float* a_row = activation + m * K;
    const block_q4_0* w_col = weight_blocks + n * num_blocks;

    // Process each block
    for (int b = 0; b < num_blocks; ++b) {
        // Extract weight scale (FP16) for this block
        float d_w = fp16_to_fp32(w_col[b].d);

        // Load activation values for this block
        float a_vals[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_vals[i] = a_row[b * 32 + i];
        }

        // Unpack weights to INT8 for dp4a
        int8_t w_q[32];
        const uint8_t* w_packed = w_col[b].qs;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            w_q[i] = (w_packed[i] & 0x0F) - 8;
            w_q[i + 16] = ((w_packed[i] >> 4) & 0x0F) - 8;
        }

        // Quantize activation to INT8 for dp4a
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
        }
        float d_a = a_max / 127.0f;
        if (d_a < 1e-6f) d_a = 1.0f;

        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = __float2int_rn(a_vals[i] / d_a);
        }

        // Compute dot product using __dp4a
        int sumi = 0;
        const int* a_q_ptr = reinterpret_cast<const int*>(a_q);
        const int* w_q_ptr = reinterpret_cast<const int*>(w_q);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sumi = __dp4a(a_q_ptr[i], w_q_ptr[i], sumi);
        }

        sum += d_w * d_a * sumi;
    }

    output[m * N + n] = sum;
}

// Optimized kernel for large M
// Uses larger tiles and better memory access patterns
constexpr int TILE_M_LARGE = 16;
constexpr int TILE_N_LARGE = 32;
constexpr int THREADS_PER_BLOCK = 512;

__global__ void q4_0_gemm_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Block and thread indices
    const int tid = threadIdx.x;

    // Tile indices
    const int tile_m_base = blockIdx.y * TILE_M_LARGE;
    const int tile_n_base = blockIdx.x * TILE_N_LARGE;

    // Thread position within tile (row-major)
    const int m_rel = tid / TILE_N_LARGE;  // 0-15
    const int n_rel = tid % TILE_N_LARGE;  // 0-31

    const int m = tile_m_base + m_rel;
    const int n = tile_n_base + n_rel;

    if (m >= M || n >= N) return;

    // Shared memory for weight tiles
    // Each iteration loads TILE_N weights, each is 18 bytes
    __shared__ float s_weight_scale[TILE_N_LARGE];
    __shared__ uint8_t s_weight_packed[TILE_N_LARGE * 16];

    // Initialize shared memory to avoid uninitialized reads
    if (tid < TILE_N_LARGE) {
        s_weight_scale[tid] = 0.0f;
    }
    #pragma unroll
    for (int i = tid; i < TILE_N_LARGE * 16; i += THREADS_PER_BLOCK) {
        s_weight_packed[i] = 0;
    }
    __syncthreads();

    float sum = 0.0f;

    const float* a_row = activation + m * K;

    // Process each K-block
    for (int b = 0; b < num_blocks; ++b) {
        // Load activation values for this block
        float a_vals[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_vals[i] = a_row[b * 32 + i];
        }

        // Load weights cooperatively into shared memory
        // Each thread loads one weight element (or skips if out of bounds)
        const int wn = n_rel;  // This thread loads weight for column n_rel in the tile

        if (tile_n_base + wn < N) {
            const block_q4_0* w_src = weight_blocks + (tile_n_base + wn) * num_blocks + b;
            // Load scale
            if (wn < TILE_N_LARGE) {
                uint16_t scale_bits = w_src->d;
                s_weight_scale[wn] = fp16_to_fp32(scale_bits);
            }
            // Load packed bytes
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (wn * 16 + i < TILE_N_LARGE * 16) {
                    s_weight_packed[wn * 16 + i] = w_src->qs[i];
                }
            }
        }

        __syncthreads();

        // Get weight for this thread's output element
        const float d_w = s_weight_scale[n_rel];
        const uint8_t* w_packed = s_weight_packed + n_rel * 16;

        // Unpack weight to INT8
        int8_t w_q[32];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            w_q[i] = (w_packed[i] & 0x0F) - 8;
            w_q[i + 16] = ((w_packed[i] >> 4) & 0x0F) - 8;
        }

        // Quantize activation to INT8
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
        }
        float d_a = a_max / 127.0f;
        if (d_a < 1e-6f) d_a = 1.0f;

        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = __float2int_rn(a_vals[i] / d_a);
        }

        // Compute dot product using __dp4a
        int sumi = 0;
        const int* a_q_ptr = reinterpret_cast<const int*>(a_q);
        const int* w_q_ptr = reinterpret_cast<const int*>(w_q);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sumi = __dp4a(a_q_ptr[i], w_q_ptr[i], sumi);
        }

        sum += d_w * d_a * sumi;

        __syncthreads();
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const float* a_ptr = activation.data_ptr<float>();
    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    float* o_ptr = output.data_ptr<float>();

    // Choose kernel based on M
    if (M <= 8) {
        // Small M: use 32-thread blocks, 8 columns per block
        int num_threads = 32;
        dim3 grid((N + 8 - 1) / 8, (M + 4 - 1) / 4);
        q4_0_gemm_small_m_kernel<<<grid, num_threads>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    } else {
        // Large M: use 512-thread blocks, 32x16 tiles
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((N + TILE_N_LARGE - 1) / TILE_N_LARGE, (M + TILE_M_LARGE - 1) / TILE_M_LARGE);
        q4_0_gemm_large_m_kernel<<<grid, block>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
