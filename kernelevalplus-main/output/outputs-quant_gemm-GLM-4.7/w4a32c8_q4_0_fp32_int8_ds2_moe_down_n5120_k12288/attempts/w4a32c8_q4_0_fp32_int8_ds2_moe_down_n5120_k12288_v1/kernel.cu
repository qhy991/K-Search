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

// Device function to compute dot product between one activation row and one weight row
__device__ float vec_dot_q4_0_fp32(
    const float* __restrict__ activation_row,  // M x K, FP32
    const uint8_t* __restrict__ weight_q4,     // Q4_0 quantized weight blocks (uint8 storage)
    int K
) {
    float sum = 0.0f;

    // Process K elements in blocks of 32
    const int num_blocks = K / 32;

    int weight_offset = 0;

    for (int block = 0; block < num_blocks; ++block) {
        // Read scale from Q4_0 block (first 2 bytes are FP16 scale)
        uint16_t scale_u16;
        memcpy(&scale_u16, &weight_q4[weight_offset], sizeof(uint16_t));
        float d_w = fp16_to_fp32(scale_u16);

        // Read packed 4-bit data (next 16 bytes)
        // Each byte: low nibble = q[i], high nibble = q[i+16]
        const uint8_t* packed = &weight_q4[weight_offset + 2];

        // Unpack and compute dot product
        // After unpacking:
        //   w_low[0..15] = weight positions 0..15
        //   w_high[0..15] = weight positions 16..31
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte_val = packed[i];
            int q_low = byte_val & 0x0F;           // q[i] for weight position i (0-15)
            int q_high = (byte_val >> 4) & 0x0F;   // q[i+16] for weight position i+16 (16-31)

            // Q4_0 dequantization: val = scale × (q - 8)
            float w_low = d_w * (q_low - 8);       // weight value at position i
            float w_high = d_w * (q_high - 8);     // weight value at position i+16

            int act_idx = block * 32 + i;

            // activation[block*32 + i] * weight[i]   (i = 0..15)
            sum += activation_row[act_idx] * w_low;
            // activation[block*32 + i + 16] * weight[i+16]   (i = 0..15)
            sum += activation_row[act_idx + 16] * w_high;
        }

        weight_offset += 18;  // 2 bytes scale + 16 bytes packed data
    }

    return sum;
}

__global__ void quant_gemm_kernel(
    const uint8_t* __restrict__ weight,     // N x (K/32) blocks, Q4_0 format (18 bytes/block)
    const float* __restrict__ activation,  // M x K
    float* __restrict__ output,            // M x N
    int M, int N, int K
) {
    // Each thread block computes one row of output
    int m = blockIdx.x;
    if (m >= M) return;

    // Each thread in the block computes a subset of N elements
    int n_start = threadIdx.x;
    int stride = blockDim.x;

    // Q4_0 weight format: each row has K/32 blocks, each block is 18 bytes
    const int blocks_per_row = K / 32;
    const int bytes_per_weight_row = blocks_per_row * 18;

    const float* act_row = &activation[m * K];
    float* out_row = &output[m * N];

    for (int n = n_start; n < N; n += stride) {
        // Get the weight row for output n
        const uint8_t* weight_row = &weight[n * bytes_per_weight_row];

        // Compute dot product
        out_row[n] = vec_dot_q4_0_fp32(act_row, weight_row, K);
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

    // For memory-bound kernels, use optimal thread count per block
    // RTX 4090 has 128 SMs, we want good occupancy
    const int threads_per_block = 256;
    const dim3 blocks(M);
    const dim3 threads(threads_per_block);

    quant_gemm_kernel<<<blocks, threads>>>(
        weight_q4.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM (Q4_0 weights, FP32 activation)");
}
