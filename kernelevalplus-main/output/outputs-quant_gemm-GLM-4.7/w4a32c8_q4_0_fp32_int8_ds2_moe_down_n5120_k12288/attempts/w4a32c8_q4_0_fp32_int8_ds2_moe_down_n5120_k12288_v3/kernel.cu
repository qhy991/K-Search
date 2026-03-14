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

// Fast Q4_0 dot product using inline dequantization
__device__ __forceinline__ float vec_dot_q4_0_fast(
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

        // Process 16 bytes of packed 4-bit data
        const uint8_t* qs = &wq[w_off + 2];

        // Unrolled loop for better throughput
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            // Load 2 bytes at once
            uint16_t packed16 = *reinterpret_cast<const uint16_t*>(&qs[i * 2]);

            // Extract 4 4-bit values
            // Byte layout: [q0, q1] where q0 = low_nibble, q1 = high_nibble
            // After loading 2 bytes: [q0_lo, q0_hi, q1_lo, q1_hi]
            // q0_lo = w_pos[0], q0_hi = w_pos[16]
            // q1_lo = w_pos[1], q1_hi = w_pos[17]
            int q0_lo = packed16 & 0x0F;
            int q0_hi = (packed16 >> 4) & 0x0F;
            int q1_lo = (packed16 >> 8) & 0x0F;
            int q1_hi = (packed16 >> 12) & 0x0F;

            // Dequantize: val = scale * (q - 8)
            float w0_lo = scale * (q0_lo - 8);
            float w0_hi = scale * (q0_hi - 8);
            float w1_lo = scale * (q1_lo - 8);
            float w1_hi = scale * (q1_hi - 8);

            // Dot with activation
            // act[a_off + 0] with w_pos[0]
            // act[a_off + 16] with w_pos[16]
            // act[a_off + 1] with w_pos[1]
            // act[a_off + 17] with w_pos[17]
            sum += act[a_off + i * 2] * w0_lo;
            sum += act[a_off + i * 2 + 16] * w0_hi;
            sum += act[a_off + i * 2 + 1] * w1_lo;
            sum += act[a_off + i * 2 + 17] * w1_hi;
        }

        w_off += 18;
        a_off += 32;
    }

    return sum;
}

// Kernel for small M (memory-bound): maximize threads per row
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

    output[m * N + n] = vec_dot_q4_0_fast(act_row, weight_row, K);
}

// Kernel for large M (compute-bound): 2D tiling
template<int TILE_N>
__global__ void gemm_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each block computes TILE_M x TILE_N output elements
    const int TILE_M = 4;

    int m_base = blockIdx.y * TILE_M;
    int n_base = blockIdx.x * TILE_N;

    int m_local = threadIdx.y;
    int n_local = threadIdx.x;

    int m = m_base + m_local;
    int n = n_base + n_local;

    if (m >= M || n >= N) return;

    const int bytes_per_weight_row = (K / 32) * 18;
    const float* act_row = &activation[m * K];
    const uint8_t* weight_row = &weight[n * bytes_per_weight_row];

    output[m * N + n] = vec_dot_q4_0_fast(act_row, weight_row, K);
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
        // Memory-bound: use 1D kernel with max threads
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
        // Compute-bound: use 2D kernel for better occupancy
        const int TILE_N = 32;
        const int TILE_M = 4;
        const dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        const dim3 threads(TILE_N, TILE_M);

        if (TILE_N == 32) {
            gemm_large_m<32><<<blocks, threads>>>(
                weight_q4.data_ptr<uint8_t>(),
                activation.data_ptr<float>(),
                output.data_ptr<float>(),
                M, N, K
            );
        }
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM (Q4_0 weights, FP32 activation) - Optimized v3");
}
