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

// BLOCK_Q4_0 format:
// - 2 bytes: FP16 scale (d)
// - 16 bytes: packed 4-bit values
//   Each byte contains 2 4-bit values: low nibble = q[i], high nibble = q[i+16]
// - Total: 18 bytes per block (32 values)
//
// Q4_0 encoding: q = round(val / scale + 8), q ∈ [0, 15]
// Q4_0 decoding: val = scale × (q - 8)

// Helper function to read FP16 as float32
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Optimized dot product with vectorized loads
__device__ __forceinline__ float vec_dot_q4_0_fp32_optimized(
    const float* __restrict__ activation_row,
    const uint8_t* __restrict__ weight_q4,
    int K
) {
    float sum = 0.0f;
    const int num_blocks = K / 32;

    // Process 2 blocks at a time for better ILP
    int block = 0;
    for (; block + 1 < num_blocks; block += 2) {
        // Block 0
        const uint8_t* packed0 = &weight_q4[block * 18];
        uint16_t scale_u16_0;
        memcpy(&scale_u16_0, packed0, sizeof(uint16_t));
        float d_w0 = fp16_to_fp32(scale_u16_0);
        const uint8_t* data0 = packed0 + 2;

        // Block 1
        const uint8_t* packed1 = &weight_q4[(block + 1) * 18];
        uint16_t scale_u16_1;
        memcpy(&scale_u16_1, packed1, sizeof(uint16_t));
        float d_w1 = fp16_to_fp32(scale_u16_1);
        const uint8_t* data1 = packed1 + 2;

        int act_base0 = block * 32;
        int act_base1 = (block + 1) * 32;

        // Process both blocks
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            // Block 0
            uint8_t byte0 = data0[i];
            float w_low0 = d_w0 * ((byte0 & 0x0F) - 8);
            float w_high0 = d_w0 * (((byte0 >> 4) & 0x0F) - 8);
            sum += activation_row[act_base0 + i] * w_low0;
            sum += activation_row[act_base0 + i + 16] * w_high0;

            // Block 1
            uint8_t byte1 = data1[i];
            float w_low1 = d_w1 * ((byte1 & 0x0F) - 8);
            float w_high1 = d_w1 * (((byte1 >> 4) & 0x0F) - 8);
            sum += activation_row[act_base1 + i] * w_low1;
            sum += activation_row[act_base1 + i + 16] * w_high1;
        }
    }

    // Handle remaining block
    if (block < num_blocks) {
        const uint8_t* packed = &weight_q4[block * 18];
        uint16_t scale_u16;
        memcpy(&scale_u16, packed, sizeof(uint16_t));
        float d_w = fp16_to_fp32(scale_u16);
        const uint8_t* data = packed + 2;
        int act_base = block * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = data[i];
            float w_low = d_w * ((byte & 0x0F) - 8);
            float w_high = d_w * (((byte >> 4) & 0x0F) - 8);
            sum += activation_row[act_base + i] * w_low;
            sum += activation_row[act_base + i + 16] * w_high;
        }
    }

    return sum;
}

__global__ void quant_gemm_kernel_optimized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.x;
    if (m >= M) return;

    int n_start = threadIdx.x;
    int stride = blockDim.x;

    const int blocks_per_row = K / 32;
    const int bytes_per_weight_row = blocks_per_row * 18;

    const float* act_row = &activation[m * K];
    float* out_row = &output[m * N];

    for (int n = n_start; n < N; n += stride) {
        const uint8_t* weight_row = &weight[n * bytes_per_weight_row];
        out_row[n] = vec_dot_q4_0_fp32_optimized(act_row, weight_row, K);
    }
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

    // Adaptive thread configuration based on M
    int threads_per_block;
    if (M == 1) {
        // For single batch, maximize threads per row
        threads_per_block = 512;
    } else if (M < 16) {
        threads_per_block = 256;
    } else {
        threads_per_block = 128;
    }

    const dim3 blocks(M);
    const dim3 threads(threads_per_block);

    quant_gemm_kernel_optimized<<<blocks, threads>>>(
        weight_q4.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM (Q4_0 weights, FP32 activation) - Optimized v2");
}
