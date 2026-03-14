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

// Simple dot product (best for large M)
__device__ __forceinline__ float vec_dot_simple(
    const float* __restrict__ act,
    const uint8_t* __restrict__ wq,
    int K
) {
    float sum = 0.0f;
    const int num_blocks = K / 32;
    int w_off = 0;
    int a_off = 0;

    for (int b = 0; b < num_blocks; ++b) {
        uint16_t scale16;
        memcpy(&scale16, &wq[w_off], 2);
        float scale = fp16_to_fp32(scale16);
        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[w_off + 2]);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t packed = qs16[i];
            float w0 = scale * (static_cast<int8_t>(packed & 0x0F) - 8);
            float w1 = scale * (static_cast<int8_t>((packed >> 4) & 0x0F) - 8);
            float w2 = scale * (static_cast<int8_t>((packed >> 8) & 0x0F) - 8);
            float w3 = scale * (static_cast<int8_t>((packed >> 12) & 0x0F) - 8);
            int base = a_off + i * 2;
            sum += act[base] * w0;
            sum += act[base + 16] * w1;
            sum += act[base + 1] * w2;
            sum += act[base + 17] * w3;
        }
        w_off += 18;
        a_off += 32;
    }
    return sum;
}

// 4-accumulator dot product (best for small M)
__device__ __forceinline__ float vec_dot_4acc(
    const float* __restrict__ act,
    const uint8_t* __restrict__ wq,
    int K
) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    const int num_blocks = K / 32;
    int w_off = 0;
    int a_off = 0;

    const int blocks_per_iter = 4;
    for (int iter = 0; iter < num_blocks / blocks_per_iter; ++iter) {
        // Block 0
        {
            uint16_t scale16;
            memcpy(&scale16, &wq[w_off], 2);
            float scale = fp16_to_fp32(scale16);
            const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[w_off + 2]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint16_t packed = qs16[i];
                float w0 = scale * (static_cast<int8_t>(packed & 0x0F) - 8);
                float w1 = scale * (static_cast<int8_t>((packed >> 4) & 0x0F) - 8);
                float w2 = scale * (static_cast<int8_t>((packed >> 8) & 0x0F) - 8);
                float w3 = scale * (static_cast<int8_t>((packed >> 12) & 0x0F) - 8);
                int base = a_off + i * 2;
                sum0 += act[base] * w0;
                sum1 += act[base + 16] * w1;
                sum2 += act[base + 1] * w2;
                sum3 += act[base + 17] * w3;
            }
            w_off += 18;
            a_off += 32;
        }
        // Block 1
        {
            uint16_t scale16;
            memcpy(&scale16, &wq[w_off + 18], 2);
            float scale = fp16_to_fp32(scale16);
            const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[w_off + 20]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint16_t packed = qs16[i];
                float w0 = scale * (static_cast<int8_t>(packed & 0x0F) - 8);
                float w1 = scale * (static_cast<int8_t>((packed >> 4) & 0x0F) - 8);
                float w2 = scale * (static_cast<int8_t>((packed >> 8) & 0x0F) - 8);
                float w3 = scale * (static_cast<int8_t>((packed >> 12) & 0x0F) - 8);
                int base = a_off + 32 + i * 2;
                sum0 += act[base] * w0;
                sum1 += act[base + 16] * w1;
                sum2 += act[base + 1] * w2;
                sum3 += act[base + 17] * w3;
            }
            w_off += 18;
            a_off += 32;
        }
        // Block 2
        {
            uint16_t scale16;
            memcpy(&scale16, &wq[w_off + 36], 2);
            float scale = fp16_to_fp32(scale16);
            const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[w_off + 38]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint16_t packed = qs16[i];
                float w0 = scale * (static_cast<int8_t>(packed & 0x0F) - 8);
                float w1 = scale * (static_cast<int8_t>((packed >> 4) & 0x0F) - 8);
                float w2 = scale * (static_cast<int8_t>((packed >> 8) & 0x0F) - 8);
                float w3 = scale * (static_cast<int8_t>((packed >> 12) & 0x0F) - 8);
                int base = a_off + 64 + i * 2;
                sum0 += act[base] * w0;
                sum1 += act[base + 16] * w1;
                sum2 += act[base + 1] * w2;
                sum3 += act[base + 17] * w3;
            }
            w_off += 18;
            a_off += 32;
        }
        // Block 3
        {
            uint16_t scale16;
            memcpy(&scale16, &wq[w_off + 54], 2);
            float scale = fp16_to_fp32(scale16);
            const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[w_off + 56]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint16_t packed = qs16[i];
                float w0 = scale * (static_cast<int8_t>(packed & 0x0F) - 8);
                float w1 = scale * (static_cast<int8_t>((packed >> 4) & 0x0F) - 8);
                float w2 = scale * (static_cast<int8_t>((packed >> 8) & 0x0F) - 8);
                float w3 = scale * (static_cast<int8_t>((packed >> 12) & 0x0F) - 8);
                int base = a_off + 96 + i * 2;
                sum0 += act[base] * w0;
                sum1 += act[base + 16] * w1;
                sum2 += act[base + 1] * w2;
                sum3 += act[base + 17] * w3;
            }
            w_off += 18;
            a_off += 32;
        }
    }

    // Handle remaining blocks
    for (int b = (num_blocks / blocks_per_iter) * blocks_per_iter; b < num_blocks; ++b) {
        uint16_t scale16;
        memcpy(&scale16, &wq[b * 18], 2);
        float scale = fp16_to_fp32(scale16);
        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(&wq[b * 18 + 2]);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t packed = qs16[i];
            float w0 = scale * (static_cast<int8_t>(packed & 0x0F) - 8);
            float w1 = scale * (static_cast<int8_t>((packed >> 4) & 0x0F) - 8);
            float w2 = scale * (static_cast<int8_t>((packed >> 8) & 0x0F) - 8);
            float w3 = scale * (static_cast<int8_t>((packed >> 12) & 0x0F) - 8);
            int base = b * 32 + i * 2;
            sum0 += act[base] * w0;
            sum1 += act[base + 16] * w1;
            sum2 += act[base + 1] * w2;
            sum3 += act[base + 17] * w3;
        }
    }
    return sum0 + sum1 + sum2 + sum3;
}

// M <= 8: 4-acc approach
__global__ __launch_bounds__(1024, 1)
void kernel_small(
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
    output[m * N + n] = vec_dot_4acc(act, w, K);
}

// M > 8: Simple approach
__global__ void kernel_large(
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
    output[m * N + n] = vec_dot_simple(act, w, K);
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

    // Combined: 4-acc for M <= 8, simple for M > 8
    if (M <= 8) {
        const int threads = (M <= 2) ? 1024 : 512;
        const dim3 blocks((N + threads - 1) / threads, M);
        kernel_small<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else {
        const int TILE_N = 128;
        const int TILE_M = 8;
        const dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        const dim3 threads(TILE_N, TILE_M);
        kernel_large<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM - v16 Combined Best");
}
