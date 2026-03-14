#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q8_0 block quantization format (34 bytes per block)
typedef struct {
    uint16_t d;     // FP16 scale (2 bytes)
    int8_t qs[32];  // Quantized values (32 bytes)
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block must be 34 bytes");

// Device function to safely read FP16 as float32
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * Kernel with shared memory optimization for small batches (M <= 8)
 * Loads activation row into shared memory to reduce global memory traffic
 */
__global__ void q8_0_gemm_shared_kernel(
    const uint8_t* __restrict__ weight_q8,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each block processes one M row
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const int num_blocks = K / 32;

    // Shared memory for activation row
    __shared__ float s_act[1536];  // K = 1536

    // Cooperatively load activation row into shared memory
    int threads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < K; i += threads) {
        s_act[i] = activation[m * K + i];
    }
    __syncthreads();

    float acc = 0.0f;

    // Compute using shared memory activation
    for (int k_block = 0; k_block < num_blocks; ++k_block) {
        const block_q8_0* w_block = reinterpret_cast<const block_q8_0*>(
            &weight_q8[n * num_blocks * 34 + k_block * 34]
        );

        float d_w = read_half_as_float(w_block->d);

        float sum = 0.0f;
        for (int j = 0; j < 32; ++j) {
            sum += s_act[k_block * 32 + j] * static_cast<float>(w_block->qs[j]);
        }
        acc += sum * d_w;
    }

    output[m * N + n] = acc;
}

/**
 * Standard kernel for larger batches (M > 8)
 * Direct global memory access (better for large M)
 */
template <int N_PER_THREAD>
__global__ void q8_0_gemm_standard_kernel(
    const uint8_t* __restrict__ weight_q8,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M) return;

    int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * N_PER_THREAD;

    const int num_blocks = K / 32;

    float acc[N_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    for (int k_block = 0; k_block < num_blocks; ++k_block) {
        const float* act_block = &activation[m * K + k_block * 32];

        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; ++i) {
            int n = n_base + i;
            if (n < N) {
                const block_q8_0* w_block = reinterpret_cast<const block_q8_0*>(
                    &weight_q8[n * num_blocks * 34 + k_block * 34]
                );

                float d_w = read_half_as_float(w_block->d);

                float sum = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    sum += act_block[j] * static_cast<float>(w_block->qs[j]);
                }
                acc[i] += sum * d_w;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        int n = n_base + i;
        if (n < N) {
            output[m * N + n] = acc[i];
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.scalar_type() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* weight_q8 = weight.data_ptr<uint8_t>();
    const float* act = activation.data_ptr<float>();
    float* out = output.data_ptr<float>();

    if (M <= 8) {
        // Shared memory kernel for small batches
        const int threads_x = 256;
        const int threads_y = 1;

        dim3 block(threads_x, threads_y);
        dim3 grid((N + threads_x - 1) / threads_x, M);

        q8_0_gemm_shared_kernel<<<grid, block>>>(weight_q8, act, out, M, N, K);
    } else {
        // Standard kernel for larger batches
        const int threads_x = 16;
        const int threads_y = 16;
        const int N_PER_THREAD = 2;

        dim3 block(threads_x, threads_y);
        dim3 grid((N + threads_x * N_PER_THREAD - 1) / (threads_x * N_PER_THREAD),
                  (M + threads_y - 1) / threads_y);

        q8_0_gemm_standard_kernel<N_PER_THREAD><<<grid, block>>>(weight_q8, act, out, M, N, K);
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM with Shared Memory (W8A32C8 Q8_0)");
}
