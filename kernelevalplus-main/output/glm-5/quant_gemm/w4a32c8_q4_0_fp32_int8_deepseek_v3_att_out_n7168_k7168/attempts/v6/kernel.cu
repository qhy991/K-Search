#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure (llama.cpp compatible)
struct block_q4_0 {
    uint16_t d;      // FP16 scale
    uint8_t qs[16];  // 32 packed quaternions
};

__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Ultra-compact kernel for M=1
__global__ void q4_0_gemm_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    // Thread ID = output column n
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.0f;
    const block_q4_0* w_col = weight_blocks + n * num_blocks;
    const float* a_row = activation;  // M=1, so row 0

    // Process each block
    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_0& wb = w_col[b];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = wb.qs[i];
            int q0 = packed & 0x0F;
            int q1 = (packed >> 4) & 0x0F;

            float a0 = a_row[k_base + i];
            float a1 = a_row[k_base + i + 16];

            sum += a0 * d_w * (q0 - 8);
            sum += a1 * d_w * (q1 - 8);
        }
    }

    output[n] = sum;  // M=1
}

// Optimized kernel for small M
constexpr int TILE_M_SMALL = 4;
constexpr int TILE_N_SMALL = 32;

__global__ void q4_0_gemm_small_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    const int tid = threadIdx.x;
    const int m = blockIdx.y * TILE_M_SMALL + tid / TILE_N_SMALL;
    const int n = blockIdx.x * TILE_N_SMALL + tid % TILE_N_SMALL;

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    const float* a_row = activation + m * K;
    const block_q4_0* w_col = weight_blocks + n * num_blocks;

    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_0& wb = w_col[b];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = wb.qs[i];
            int q0 = packed & 0x0F;
            int q1 = (packed >> 4) & 0x0F;

            float a0 = a_row[k_base + i];
            float a1 = a_row[k_base + i + 16];

            sum += a0 * d_w * (q0 - 8);
            sum += a1 * d_w * (q1 - 8);
        }
    }

    output[m * N + n] = sum;
}

// Optimized kernel for medium M
constexpr int TILE_M_MEDIUM = 8;
constexpr int TILE_N_MEDIUM = 32;

__global__ void q4_0_gemm_medium_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    const int tid = threadIdx.x;
    const int m = blockIdx.y * TILE_M_MEDIUM + tid / TILE_N_MEDIUM;
    const int n = blockIdx.x * TILE_N_MEDIUM + tid % TILE_N_MEDIUM;

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    const float* a_row = activation + m * K;
    const block_q4_0* w_col = weight_blocks + n * num_blocks;

    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_0& wb = w_col[b];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = wb.qs[i];
            int q0 = packed & 0x0F;
            int q1 = (packed >> 4) & 0x0F;

            float a0 = a_row[k_base + i];
            float a1 = a_row[k_base + i + 16];

            sum += a0 * d_w * (q0 - 8);
            sum += a1 * d_w * (q1 - 8);
        }
    }

    output[m * N + n] = sum;
}

// Optimized kernel for large M
constexpr int TILE_M_LARGE = 16;
constexpr int TILE_N_LARGE = 32;

__global__ void q4_0_gemm_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    const int tid = threadIdx.x;
    const int m = blockIdx.y * TILE_M_LARGE + tid / TILE_N_LARGE;
    const int n = blockIdx.x * TILE_N_LARGE + tid % TILE_N_LARGE;

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    const float* a_row = activation + m * K;
    const block_q4_0* w_col = weight_blocks + n * num_blocks;

    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_0& wb = w_col[b];
        float d_w = fp16_to_fp32(wb.d);
        int k_base = b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = wb.qs[i];
            int q0 = packed & 0x0F;
            int q1 = (packed >> 4) & 0x0F;

            float a0 = a_row[k_base + i];
            float a1 = a_row[k_base + i + 16];

            sum += a0 * d_w * (q0 - 8);
            sum += a1 * d_w * (q1 - 8);
        }
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

    // Strategy dispatch based on M for optimal tile sizes
    if (M == 1) {
        int num_threads = 256;
        int num_blocks = (N + num_threads - 1) / num_threads;
        q4_0_gemm_m1_kernel<<<num_blocks, num_threads>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    } else if (M <= 4) {
        dim3 block(TILE_N_SMALL * TILE_M_SMALL);
        dim3 grid((N + TILE_N_SMALL - 1) / TILE_N_SMALL, (M + TILE_M_SMALL - 1) / TILE_M_SMALL);
        q4_0_gemm_small_m_kernel<<<grid, block>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    } else if (M <= 8) {
        dim3 block(TILE_N_MEDIUM * TILE_M_MEDIUM);
        dim3 grid((N + TILE_N_MEDIUM - 1) / TILE_N_MEDIUM, (M + TILE_M_MEDIUM - 1) / TILE_M_MEDIUM);
        q4_0_gemm_medium_m_kernel<<<grid, block>>>(
            w_ptr, a_ptr, o_ptr, M, N, K
        );
    } else {
        dim3 block(TILE_N_LARGE * TILE_M_LARGE);
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
