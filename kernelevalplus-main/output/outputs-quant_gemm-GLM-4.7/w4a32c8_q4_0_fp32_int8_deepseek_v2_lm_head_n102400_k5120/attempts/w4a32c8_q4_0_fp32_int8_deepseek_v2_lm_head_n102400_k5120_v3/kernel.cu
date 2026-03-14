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

// Optimized: Load and unpack 32 Q4_0 values into FP32 in shared memory
// This allows reuse of weight blocks across multiple output computations
__device__ __forceinline__ void unpack_q4_0_to_fp32(
    const uint8_t* __restrict__ weight_block,
    float* __restrict__ output_weights  // Must have 32 elements
) {
    // Load scale
    uint16_t scale_u16;
    memcpy(&scale_u16, weight_block, 2);
    const float scale = half_to_float(scale_u16);

    // Get packed data
    const uint8_t* packed = weight_block + 2;

    // Unpack all 32 values: positions 0-15 = low nibbles, 16-31 = high nibbles
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        output_weights[i] = scale * (float)((packed[i] & 0x0F) - 8);
        output_weights[i + 16] = scale * (float)(((packed[i] >> 4) & 0x0F) - 8);
    }
}

// Compute dot product using pre-unpacked FP32 weights
__device__ __forceinline__ float dot_product_fp32(
    const float* __restrict__ weights,  // 32 elements
    const float* __restrict__ activation
) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        sum += weights[i] * activation[i];
    }
    return sum;
}

// Kernel that tiles K dimension and caches weight blocks in shared memory
__global__ void w4a32c8_q4_0_gemm_kernel_v3_tiled(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each block processes one M (batch) and a chunk of N
    const int m_idx = blockIdx.x;
    const int n_chunk_base = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_stride = gridDim.y * blockDim.y;

    if (m_idx >= M) return;

    // Shared memory for: activation tile + weight blocks for multiple N values
    // Each thread in y-dim brings one weight block
    extern __shared__ float s_mem[];
    float* s_activation = s_mem;
    float* s_weights = s_activation + K;  // Rest of shared memory for weights

    const int num_blocks_k = K / 32;
    const int weights_per_thread = blockDim.y;  // Each y-thread handles this many N values

    // Load activation row cooperatively
    const float* activation_row = activation + m_idx * K;
    const int load_tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int load_stride = blockDim.x * blockDim.y;

    for (int i = load_tid; i < K; i += load_stride) {
        s_activation[i] = activation_row[i];
    }

    __syncthreads();

    // Each y-thread processes different N values
    // x-threads cooperatively load and unpack weight blocks
    for (int n_idx = n_chunk_base; n_idx < N; n_idx += n_stride) {
        float acc = 0.0f;

        for (int kb = 0; kb < num_blocks_k; ++kb) {
            // Load weight block for this N
            if (threadIdx.x == 0) {
                const uint8_t* w_block = weight + (n_idx * num_blocks_k + kb) * 18;
                unpack_q4_0_to_fp32(w_block, s_weights + threadIdx.y * 32);
            }
            __syncthreads();

            // Each y-thread computes its output
            const float* act_block = s_activation + kb * 32;
            const float* w_block = s_weights + threadIdx.y * 32;
            acc += dot_product_fp32(w_block, act_block);

            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[m_idx * N + n_idx] = acc;
        }
    }
}

// Simplified kernel: better memory coalescing with vectorized loads
__global__ void w4a32c8_q4_0_gemm_kernel_v3_vectorized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int n_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (m_idx >= M || n_idx >= N) return;

    // Each thread loads its activation values directly (optimized caching)
    const float* activation_row = activation + m_idx * K;
    const int num_blocks_k = K / 32;

    float acc = 0.0f;

    // Process K in blocks of 32
    for (int kb = 0; kb < num_blocks_k; ++kb) {
        // Load weight block
        const uint8_t* w_block = weight + (n_idx * num_blocks_k + kb) * 18;

        // Unpack and accumulate
        uint16_t scale_u16;
        memcpy(&scale_u16, w_block, 2);
        const float scale = half_to_float(scale_u16);
        const uint8_t* packed = w_block + 2;

        // Process low nibbles (positions 0-15)
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float w = scale * (float)((packed[i] & 0x0F) - 8);
            acc += w * activation_row[kb * 32 + i];
        }

        // Process high nibbles (positions 16-31)
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float w = scale * (float)(((packed[i] >> 4) & 0x0F) - 8);
            acc += w * activation_row[kb * 32 + i + 16];
        }
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

    // For very large N (102400), use 2D grid for better parallelism
    const int threads_x = 1;
    const int threads_y = 256;
    const int n_chunks = (N + threads_y - 1) / threads_y;

    dim3 grid(M, min(n_chunks, 512));
    dim3 block(threads_x, threads_y);

    w4a32c8_q4_0_gemm_kernel_v3_vectorized<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v3");
}
