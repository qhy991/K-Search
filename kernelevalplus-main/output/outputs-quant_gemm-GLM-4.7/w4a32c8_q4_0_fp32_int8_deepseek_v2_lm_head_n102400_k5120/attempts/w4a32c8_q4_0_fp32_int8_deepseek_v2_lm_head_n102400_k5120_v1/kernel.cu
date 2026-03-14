#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure: scale (FP16) + 16 packed bytes (32 x 4-bit values)
// typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;
// static_assert(sizeof(block_q4_0) == 18, "");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half raw; } un;
    un.u16 = h;
    return __half2float(un.raw);
}

// Device function to compute dot product of 32 Q4_0 values with FP32 activations
// Uses llama.cpp compatible unpacking (all low nibbles first, then all high nibbles)
__device__ __forceinline__ float q4_0_dot_32(const uint8_t* weight_block, const float* activation) {
    // Load scale (first 2 bytes as FP16) - always aligned
    uint16_t scale_u16;
    memcpy(&scale_u16, weight_block, 2);
    float scale = half_to_float(scale_u16);

    // Get packed quantized data (16 bytes after scale)
    const uint8_t* packed = weight_block + 2;

    float sum = 0.0f;

    // Q4_0 packing format (llama.cpp compatible):
    // - Position 0-15: low nibbles of bytes 0-15
    // - Position 16-31: high nibbles of bytes 0-15

    // First process all 16 low nibbles (positions 0-15)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t q = packed[i] & 0x0F;  // Low nibble
        sum += scale * (float)(q - 8) * activation[i];
    }

    // Then process all 16 high nibbles (positions 16-31)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t q = (packed[i] >> 4) & 0x0F;  // High nibble
        sum += scale * (float)(q - 8) * activation[i + 16];
    }

    return sum;
}

// Main kernel for W4A32C8 Q4_0 GEMM
// Each thread block computes one output row (M dimension)
// Each thread computes multiple N output values using a loop
__global__ void w4a32c8_q4_0_gemm_kernel(
    const uint8_t* __restrict__ weight,  // [N, K/32] Q4_0 blocks, each 18 bytes
    const float* __restrict__ activation,  // [M, K] FP32
    float* __restrict__ output,  // [M, N] FP32
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;  // Which batch element
    const int n_stride = blockDim.x;
    const int n_start = threadIdx.x;

    if (m_idx >= M) return;

    // Shared memory for activation row (reused across all N computations)
    __shared__ float s_activation[5120];  // K = 5120

    // Coalesced load of activation row
    const float* activation_row = activation + m_idx * K;

    // Each thread loads a portion of the activation row
    const int load_elements = (K + blockDim.x - 1) / blockDim.x;
    const int load_start = threadIdx.x * load_elements;
    const int load_end = min(load_start + load_elements, K);

    for (int i = load_start; i < load_end; ++i) {
        s_activation[i] = activation_row[i];
    }

    __syncthreads();

    // Compute outputs for assigned N values (strided access)
    const int num_blocks_k = K / 32;  // 160 blocks for K=5120

    for (int n_idx = n_start; n_idx < N; n_idx += n_stride) {
        float acc = 0.0f;

        // Compute dot product across all K blocks
        for (int kb = 0; kb < num_blocks_k; ++kb) {
            const uint8_t* w_block = weight + (n_idx * num_blocks_k + kb) * 18;
            const float* act_block = s_activation + kb * 32;

            acc += q4_0_dot_32(w_block, act_block);
        }

        output[m_idx * N + n_idx] = acc;
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    // Use simple kernel with strided access for large N
    const int threads_per_block = 256;

    dim3 grid(M);
    dim3 block(threads_per_block);

    w4a32c8_q4_0_gemm_kernel<<<grid, block>>>(
        (const uint8_t*)weight.data_ptr(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
