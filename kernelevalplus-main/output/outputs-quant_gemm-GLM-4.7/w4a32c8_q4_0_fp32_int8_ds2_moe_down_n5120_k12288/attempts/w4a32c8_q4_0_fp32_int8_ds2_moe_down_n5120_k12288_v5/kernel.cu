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

// Ultra-optimized dot product with 16-bit vectorized loads
__device__ __forceinline__ float vec_dot_q4_0_ultra(
    const float* __restrict__ act,
    const uint8_t* __restrict__ wq,
    int K
) {
    float sum = 0.0f;
    const int num_blocks = K / 32;
    int w_off = 0;
    int a_off = 0;

    for (int b = 0; b < num_blocks; ++b) {
        // Load scale
        uint16_t scale16;
        memcpy(&scale16, &wq[w_off], 2);
        float scale = fp16_to_fp32(scale16);

        // Load packed data with uint16_t
        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[w_off + 2]);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t packed = qs16[i];
            float w0_lo = scale * ((packed & 0x0F) - 8);
            float w0_hi = scale * (((packed >> 4) & 0x0F) - 8);
            float w1_lo = scale * (((packed >> 8) & 0x0F) - 8);
            float w1_hi = scale * (((packed >> 12) & 0x0F) - 8);

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

// Strategy 1: Ultra-small M (1-2) - Maximize threads per output
__global__ __launch_bounds__(1024, 2)
void kernel_strategy1(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.y;
    if (m >= M) return;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];
    output[m * N + n] = vec_dot_q4_0_ultra(act, w, K);
}

// Strategy 2: Small M (3-16) - Balanced thread configuration
__global__ __launch_bounds__(512, 2)
void kernel_strategy2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.y;
    if (m >= M) return;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];
    output[m * N + n] = vec_dot_q4_0_ultra(act, w, K);
}

// Strategy 3: Large M (17+) - 2D grid for optimal occupancy
__global__ void kernel_strategy3(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];
    output[m * N + n] = vec_dot_q4_0_ultra(act, w, K);
}

torch::Tensor forward(
    torch::Tensor weight_q4,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be FP32");
    TORCH_CHECK(activation.device().is_cuda(), "Activation must be on CUDA device");
    TORCH_CHECK(weight_q4.device().is_cuda(), "Weight must be on CUDA device");

    auto options = torch::dtype(torch::kFloat32).device(activation.device());
    auto output = torch::empty({M, N}, options);

    auto w_ptr = weight_q4.data_ptr<uint8_t>();
    auto a_ptr = activation.data_ptr<float>();
    auto o_ptr = output.data_ptr<float>();

    // Strategy dispatch based on batch size M
    if (M <= 2) {
        // Strategy 1: Ultra-small batch - Maximize parallelism per row
        const int threads = 1024;
        const dim3 blocks((N + threads - 1) / threads, M);
        kernel_strategy1<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else if (M <= 16) {
        // Strategy 2: Small batch - Balanced configuration
        const int threads = 512;
        const dim3 blocks((N + threads - 1) / threads, M);
        kernel_strategy2<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else {
        // Strategy 3: Large batch - 2D tiling for compute-bound
        const int TILE_N = 128;
        const int TILE_M = 8;
        const dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        const dim3 threads(TILE_N, TILE_M);
        kernel_strategy3<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM - Final Combined Strategy");
}
