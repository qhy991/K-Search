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

// Combined load and dot product for better ILP
__device__ __forceinline__ float q4_0_dot_32_fused(
    const uint8_t* __restrict__ weight_block,
    const float* __restrict__ activation
) {
    uint16_t scale_u16;
    memcpy(&scale_u16, weight_block, 2);
    const float scale = half_to_float(scale_u16);
    const uint8_t* packed = weight_block + 2;

    float sum = 0.0f;

    // Process low nibbles (positions 0-15) - fully unrolled
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float w = scale * (float)((packed[i] & 0x0F) - 8);
        sum += w * activation[i];
    }

    // Process high nibbles (positions 16-31) - fully unrolled
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float w = scale * (float)(((packed[i] >> 4) & 0x0F) - 8);
        sum += w * activation[i + 16];
    }

    return sum;
}

// Kernel optimized for large batch sizes
// Uses 2D grid with direct global memory access (no shared memory)
// This reduces synchronization overhead and improves occupancy
__global__ void w4a32c8_q4_0_gemm_kernel_v4_large_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // 2D thread mapping: each thread computes one (m, n) output element
    const int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (m_idx >= M || n_idx >= N) return;

    const float* activation_row = activation + m_idx * K;
    const int num_blocks_k = K / 32;

    float acc = 0.0f;

    // Compute dot product across K dimension
    for (int kb = 0; kb < num_blocks_k; ++kb) {
        const uint8_t* w_block = weight + (n_idx * num_blocks_k + kb) * 18;
        const float* act_block = activation_row + kb * 32;
        acc += q4_0_dot_32_fused(w_block, act_block);
    }

    output[m_idx * N + n_idx] = acc;
}

// Kernel optimized for small batch sizes
// Uses shared memory to cache activation rows
__global__ void w4a32c8_q4_0_gemm_kernel_v4_small_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int n_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (m_idx >= M || n_idx >= N) return;

    // Shared memory for activation row
    __shared__ float s_activation[5120];

    // Cooperatively load activation row
    const float* activation_row = activation + m_idx * K;
    const int load_tid = threadIdx.y;
    const int load_stride = blockDim.y;

    for (int i = load_tid; i < K; i += load_stride) {
        s_activation[i] = activation_row[i];
    }

    __syncthreads();

    const int num_blocks_k = K / 32;
    float acc = 0.0f;

    for (int kb = 0; kb < num_blocks_k; ++kb) {
        const uint8_t* w_block = weight + (n_idx * num_blocks_k + kb) * 18;
        const float* act_block = s_activation + kb * 32;
        acc += q4_0_dot_32_fused(w_block, act_block);
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

    // Choose kernel based on batch size
    if (M >= 64) {
        // Large batch: use direct 2D kernel for maximum occupancy
        const int threads_x = 16;
        const int threads_y = 16;

        const int grid_x = (M + threads_x - 1) / threads_x;
        const int grid_y = (N + threads_y - 1) / threads_y;

        dim3 grid(grid_x, grid_y);
        dim3 block(threads_x, threads_y);

        w4a32c8_q4_0_gemm_kernel_v4_large_batch<<<grid, block>>>(
            (const uint8_t*)weight.data_ptr(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Small batch: use shared memory to cache activations
        const int threads_x = 1;
        const int threads_y = 256;

        const int grid_x = M;
        const int grid_y = (N + threads_y - 1) / threads_y;

        dim3 grid(grid_x, grid_y);
        dim3 block(threads_x, threads_y);

        w4a32c8_q4_0_gemm_kernel_v4_small_batch<<<grid, block>>>(
            (const uint8_t*)weight.data_ptr(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v4");
}
