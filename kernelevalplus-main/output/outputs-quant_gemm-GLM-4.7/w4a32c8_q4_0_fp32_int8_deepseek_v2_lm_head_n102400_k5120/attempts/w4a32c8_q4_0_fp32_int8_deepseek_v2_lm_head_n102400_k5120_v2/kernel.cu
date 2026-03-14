#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure: scale (FP16) + 16 packed bytes (32 x 4-bit values)
__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half raw; } un;
    un.u16 = h;
    return __half2float(un.raw);
}

// Optimized device function to compute dot product of 32 Q4_0 values with FP32 activations
__device__ __forceinline__ float q4_0_dot_32_optimized(const uint8_t* __restrict__ weight_block, const float* __restrict__ activation) {
    // Load scale (first 2 bytes as FP16)
    uint16_t scale_u16;
    memcpy(&scale_u16, weight_block, 2);
    const float scale = half_to_float(scale_u16);

    // Get packed quantized data (16 bytes after scale)
    const uint8_t* packed = weight_block + 2;

    // Process all 32 values with full unrolling
    // Q4_0 format: positions 0-15 = low nibbles, 16-31 = high nibbles
    float sum = 0.0f;

    // Low nibbles (positions 0-15)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = packed[i] & 0x0F;
        sum += scale * (float)(q - 8) * activation[i];
    }

    // High nibbles (positions 16-31)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = (packed[i] >> 4) & 0x0F;
        sum += scale * (float)(q - 8) * activation[i + 16];
    }

    return sum;
}

// Optimized kernel: each thread processes multiple N values with strided access
__global__ void w4a32c8_q4_0_gemm_kernel_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int n_start = threadIdx.x;
    const int n_stride = blockDim.x;

    if (m_idx >= M) return;

    // Shared memory for activation row (padded to avoid bank conflicts)
    __shared__ float s_activation[5120];

    // Coalesced load of activation row
    const float* activation_row = activation + m_idx * K;
    const int elements_per_thread = (K + blockDim.x - 1) / blockDim.x;
    const int load_start = threadIdx.x * elements_per_thread;
    const int load_end = min(load_start + elements_per_thread, K);

    for (int i = load_start; i < load_end; ++i) {
        s_activation[i] = activation_row[i];
    }

    __syncthreads();

    const int num_blocks_k = K / 32;

    // Process N values with strided access for better load balancing
    for (int n_idx = n_start; n_idx < N; n_idx += n_stride) {
        float acc = 0.0f;

        for (int kb = 0; kb < num_blocks_k; ++kb) {
            const uint8_t* w_block = weight + (n_idx * num_blocks_k + kb) * 18;
            const float* act_block = s_activation + kb * 32;
            acc += q4_0_dot_32_optimized(w_block, act_block);
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

    const int threads_per_block = 256;

    dim3 grid(M);
    dim3 block(threads_per_block);

    w4a32c8_q4_0_gemm_kernel_v2<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v2");
}
