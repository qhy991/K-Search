#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdint.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

// Helper function to read FP16 as float32
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Optimized dot product with 16-bit vectorized loads
__device__ __forceinline__ float vec_dot_q4_0_optimized(
    const float* __restrict__ act,
    const uint8_t* __restrict__ wq,
    int K
) {
    float sum = 0.0f;
    const int num_blocks = K / 32;

    int w_off = 0;
    int a_off = 0;

    for (int b = 0; b < num_blocks; ++b) {
        // Load scale (2 bytes FP16)
        uint16_t scale16;
        memcpy(&scale16, &wq[w_off], 2);
        float scale = fp16_to_fp32(scale16);

        // Process 16 bytes using uint16_t loads (safe alignment)
        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[w_off + 2]);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t packed16 = qs16[i];

            // Extract 4 nibbles from 2 bytes
            int q0_lo = packed16 & 0x0F;
            int q0_hi = (packed16 >> 4) & 0x0F;
            int q1_lo = (packed16 >> 8) & 0x0F;
            int q1_hi = (packed16 >> 12) & 0x0F;

            // Dequantize
            float w0_lo = scale * (q0_lo - 8);
            float w0_hi = scale * (q0_hi - 8);
            float w1_lo = scale * (q1_lo - 8);
            float w1_hi = scale * (q1_hi - 8);

            int base = a_off + i * 2;
            sum += act[base] * w0_lo;
            sum += act[base + 16] * w0_hi;
            sum += act[base + 1] * w1_lo;
            sum += act[base + 17] * w1_hi;
        }

        w_off += 18;
        a_off += 32;
    }

    return sum;
}

// Memory-bound kernel for small M
__global__ void gemm_small_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.y;
    if (m >= M) return;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int bytes_per_weight_row = (K / 32) * 18;
    const float* act_row = &activation[m * K];
    const uint8_t* weight_row = &weight[n * bytes_per_weight_row];

    output[m * N + n] = vec_dot_q4_0_optimized(act_row, weight_row, K);
}

// Compute-bound kernel with 2D grid
__global__ void gemm_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const int bytes_per_weight_row = (K / 32) * 18;
    const float* act_row = &activation[m * K];
    const uint8_t* weight_row = &weight[n * bytes_per_weight_row];

    output[m * N + n] = vec_dot_q4_0_optimized(act_row, weight_row, K);
}

torch::Tensor forward(
    torch::Tensor weight_q4,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be FP32");
    TORCH_CHECK(activation.device().is_cuda(), "Activation must be on CUDA device");
    TORCH_CHECK(weight_q4.device().is_cuda(), "Weight must be on CUDA device");

    auto options = torch::dtype(torch::kFloat32)
                      .device(activation.device())
                      .layout(torch::kStrided);

    auto output = torch::empty({M, N}, options);

    if (M <= 8) {
        // Memory-bound: 1D kernel
        const int threads_per_block = 512;
        const int blocks_x = (N + threads_per_block - 1) / threads_per_block;
        const dim3 blocks(blocks_x, M);
        const dim3 threads(threads_per_block);

        gemm_small_m<<<blocks, threads>>>(
            weight_q4.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Compute-bound: 2D kernel with larger tile size
        const int TILE_N = 64;
        const int TILE_M = 16;
        const dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        const dim3 threads(TILE_N, TILE_M);

        gemm_large_m<<<blocks, threads>>>(
            weight_q4.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM (Q4_0 weights, FP32 activation) - Optimized v4");
}
