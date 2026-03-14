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

// Naive kernel: each thread computes one output element
// Uses __dp4a for efficient dot products with dynamic activation quantization
__global__ void q4_0_gemm_naive_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = tid / N;
    const int n = tid % N;

    if (m >= M || n >= N) return;

    const int num_blocks = K / 32;

    // Accumulator
    float sum = 0.0f;

    const float* a_row = activation + m * K;
    // Cast weight to block_q4_0*
    const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(weight) + n * num_blocks;

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
        // q - 8 shifts range from [0, 15] to [-8, 7] which fits INT8
        int8_t w_q[32];
        const uint8_t* w_packed = w_col[b].qs;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            w_q[i] = (w_packed[i] & 0x0F) - 8;                    // Low nibbles
            w_q[i + 16] = ((w_packed[i] >> 4) & 0x0F) - 8;       // High nibbles
        }

        // Quantize activation to INT8 for dp4a
        // Use per-block quantization (Q8_0 style)
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
        // dp4a(a, b, c) computes: sum((int8_t*)a * (int8_t*)b) + c
        int sumi = 0;
        const int* a_q_ptr = reinterpret_cast<const int*>(a_q);
        const int* w_q_ptr = reinterpret_cast<const int*>(w_q);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sumi = __dp4a(a_q_ptr[i], w_q_ptr[i], sumi);
        }

        // Apply scaling: d_w * d_a * sumi
        sum += d_w * d_a * sumi;
    }

    output[m * N + n] = sum;
}

// Simplified tiled kernel - processes 2x4 tile with 2D thread layout
constexpr int TILE_M = 2;
constexpr int TILE_N = 4;
constexpr int BLOCK_K = 32;  // Matches Q4_0 block size

__global__ void q4_0_gemm_tiled_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // 2D thread layout
    const int tx = threadIdx.x;  // 0-3 (N dimension)
    const int ty = threadIdx.y;  // 0-1 (M dimension)

    const int tile_m_base = blockIdx.y * TILE_M;
    const int tile_n_base = blockIdx.x * TILE_N;

    const int m = tile_m_base + ty;
    const int n = tile_n_base + tx;

    if (m >= M || n >= N) return;

    // Shared memory for weight block (TILE_N weights)
    __shared__ uint8_t s_weight[TILE_N * 18];  // Each weight is 18 bytes

    float sum = 0.0f;

    const float* a_row = activation + m * K;

    // Process each K-block
    for (int b = 0; b < num_blocks; ++b) {
        // Load activation values for this block
        float a_vals[BLOCK_K];
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            a_vals[i] = a_row[b * BLOCK_K + i];
        }

        // Each thread loads one weight element (18 bytes) into shared memory
        // Thread (ty, tx) loads the weight for output column (tile_n_base + tx)
        if (tile_n_base + tx < N) {
            const block_q4_0* w_src = weight_blocks + (tile_n_base + tx) * num_blocks + b;
            // Copy entire 18-byte block to shared memory
            uint8_t* s_w_ptr = s_weight + tx * 18;
            #pragma unroll
            for (int i = 0; i < 18; ++i) {
                s_w_ptr[i] = reinterpret_cast<const uint8_t*>(w_src)[i];
            }
        }

        __syncthreads();

        // Get weight for this thread's output element
        const uint8_t* w_bytes = s_weight + tx * 18;
        // Extract scale (FP16)
        uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(w_bytes);
        float d_w = fp16_to_fp32(scale_bits);
        const uint8_t* w_packed = w_bytes + 2;

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
        for (int i = 0; i < BLOCK_K; ++i) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
        }
        float d_a = a_max / 127.0f;
        if (d_a < 1e-6f) d_a = 1.0f;

        int8_t a_q[BLOCK_K];
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
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

        // Apply scaling
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
    // Small M: use naive kernel (simpler, less shared memory overhead)
    // Large M: use tiled kernel (better memory access patterns)
    if (M <= 8) {
        int num_threads = 256;
        int num_blocks = (M * N + num_threads - 1) / num_threads;
        q4_0_gemm_naive_kernel<<<num_blocks, num_threads>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    } else {
        // Tiled kernel configuration
        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        q4_0_gemm_tiled_kernel<<<grid, block>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
