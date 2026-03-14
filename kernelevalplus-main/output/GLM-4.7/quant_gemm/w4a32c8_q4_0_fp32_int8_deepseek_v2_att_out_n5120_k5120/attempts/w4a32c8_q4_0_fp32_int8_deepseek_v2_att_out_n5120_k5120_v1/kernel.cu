#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure (llama.cpp compatible)
// Each block contains: FP16 scale (2 bytes) + 32 packed quaternions (16 bytes) = 18 bytes
struct block_q4_0 {
    uint16_t d;      // FP16 scale (delta)
    uint8_t qs[16];  // 32 packed quaternions (4-bit each, offset-8 encoded)
};

// Device function to read FP16 as FP32
__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Simple kernel: Q4_0 weight × FP32 activation
// Each thread computes one output element
__global__ void q4_0_fp32_gemm_kernel(
    const block_q4_0* __restrict__ weight_q4,   // Q4_0 weights [N, K/32] blocks
    const float* __restrict__ activation_fp32,  // FP32 activations [M, K]
    float* __restrict__ output,                 // Output [M, N]
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    // Thread mapping: blockIdx.x processes N dimension, blockIdx.y processes M dimension
    const int n = blockIdx.x;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    // Accumulator for this output element
    float acc = 0.0f;

    // Process K in blocks of 32
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load weight block
        int block_idx = n * num_blocks_k + kb;
        const block_q4_0& w_block = weight_q4[block_idx];

        // Read weight scale
        float d_w = read_half_as_float(w_block.d);

        // Unpack Q4_0 weights (llama.cpp format: low nibbles first, then high nibbles)
        // byte[i] = q[i] | (q[i+16] << 4)
        int32_t w_qs[32];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = w_block.qs[i];
            w_qs[i] = packed & 0x0F;           // Low nibble (position 0-15)
            w_qs[i + 16] = (packed >> 4) & 0x0F; // High nibble (position 16-31)
        }

        // Load activation block and compute dot product
        int k_base = kb * 32;
        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float act_val = activation_fp32[m * K + k_base + i];
            // Q4_0 decoding: value = d_w * (q - 8)
            block_sum += act_val * d_w * (w_qs[i] - 8);
        }
        acc += block_sum;
    }

    // Write result
    output[m * N + n] = acc;
}

torch::Tensor forward(
    torch::Tensor weight_q4,
    torch::Tensor activation_fp32,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight_q4.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation_fp32.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(activation_fp32.scalar_type() == torch::kFloat32, "Activation must be FP32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight_q4.device()));

    // Grid: N columns × M rows
    dim3 grid(N, M);
    dim3 block(1);  // One thread per output element

    q4_0_fp32_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const block_q4_0*>(weight_q4.data_ptr<uint8_t>()),
        activation_fp32.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
