#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

// Helper: Convert FP16 to FP32
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Optimized dot product with improved memory access pattern
__device__ __forceinline__ float vec_dot_q4_0_optimized(
    const float* __restrict__ act,
    const uint8_t* __restrict__ wq,
    int K
) {
    float sum = 0.0f;
    const int num_blocks = K / 32;

    for (int b = 0; b < num_blocks; ++b) {
        uint16_t scale16;
        memcpy(&scale16, &wq[0], 2);
        const float scale = fp16_to_fp32(scale16);

        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[2]);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint16_t packed = qs16[i];
            const float w00 = scale * ((packed & 0x0F) - 8.0f);
            const float w01 = scale * (((packed >> 4) & 0x0F) - 8.0f);
            const float w16 = scale * (((packed >> 8) & 0x0F) - 8.0f);
            const float w17 = scale * (((packed >> 12) & 0x0F) - 8.0f);

            const int base = i * 2;
            sum += act[base] * w00;
            sum += act[base + 16] * w01;
            sum += act[base + 1] * w16;
            sum += act[base + 17] * w17;
        }

        wq += 18;
        act += 32;
    }

    return sum;
}

// Strategy 1: M=1
__global__ __launch_bounds__(1024, 2)
void gemm_m1(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    if (m >= M) return;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];

    output[m * N + n] = vec_dot_q4_0_optimized(act, w, K);
}

// Strategy 2: M=2-8
__global__ __launch_bounds__(512, 2)
void gemm_small(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    if (m >= M) return;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];

    output[m * N + n] = vec_dot_q4_0_optimized(act, w, K);
}

// Strategy 3: M=9-512
__global__ void gemm_large(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];

    output[m * N + n] = vec_dot_q4_0_optimized(act, w, K);
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

    auto w_ptr = weight_q4.data_ptr<uint8_t>();
    auto a_ptr = activation.data_ptr<float>();
    auto o_ptr = output.data_ptr<float>();

    if (M == 1) {
        const int threads = 1024;
        const dim3 blocks((N + threads - 1) / threads, M);
        gemm_m1<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else if (M <= 8) {
        const int threads = 512;
        const dim3 blocks((N + threads - 1) / threads, M);
        gemm_small<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else {
        const int TILE_N = 128;
        const int TILE_M = 8;
        const dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        const dim3 threads(TILE_N, TILE_M);
        gemm_large<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM - v9 Optimized");
}
