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

// Optimized dot product with manual prefetching and ILP optimization
__device__ __forceinline__ float q4_0_dot_32_optimized(
    const uint8_t* __restrict__ weight_block,
    const float* __restrict__ activation
) {
    // Load scale
    uint16_t scale_u16;
    memcpy(&scale_u16, weight_block, 2);
    const float scale = half_to_float(scale_u16);
    const uint8_t* packed = weight_block + 2;

    float sum = 0.0f;

    // Process low nibbles (positions 0-15) - fully unrolled for ILP
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float w = scale * (float)((packed[i] & 0x0F) - 8);
        sum += w * activation[i];
    }

    // Process high nibbles (positions 16-31) - fully unrolled for ILP
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float w = scale * (float)(((packed[i] >> 4) & 0x0F) - 8);
        sum += w * activation[i + 16];
    }

    return sum;
}

// Main kernel: 2D grid for optimal parallelism
// Each thread computes one output element (m, n)
__global__ void w4a32c8_q4_0_gemm_kernel_v5(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int n_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (m_idx >= M || n_idx >= N) return;

    const float* activation_row = activation + m_idx * K;
    const int num_blocks_k = K / 32;

    float acc = 0.0f;

    // Process K dimension in blocks of 32
    for (int kb = 0; kb < num_blocks_k; ++kb) {
        const uint8_t* w_block = weight + (n_idx * num_blocks_k + kb) * 18;
        const float* act_block = activation_row + kb * 32;
        acc += q4_0_dot_32_optimized(w_block, act_block);
    }

    output[m_idx * N + n_idx] = acc;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    // Optimal block size from v3: 1 x 256
    const int threads_x = 1;
    const int threads_y = 256;

    // Calculate grid dimensions
    const int grid_x = M;
    const int grid_y = (N + threads_y - 1) / threads_y;

    dim3 grid(grid_x, grid_y);
    dim3 block(threads_x, threads_y);

    w4a32c8_q4_0_gemm_kernel_v5<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v5");
}
