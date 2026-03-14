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
 * Simplified kernel for all batch sizes
 * Uses 2D grid-stride loop for clean, correct computation
 */
template <int N_PER_THREAD>
__global__ void q8_0_gemm_kernel(
    const uint8_t* __restrict__ weight_q8,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // 2D thread mapping
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * N_PER_THREAD;

    if (m >= M) return;

    const int num_blocks = K / 32;

    // Accumulators
    float acc[N_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    // Compute across K dimension
    for (int k_block = 0; k_block < num_blocks; ++k_block) {
        // Load activation block for this row
        const float* act_block = &activation[m * K + k_block * 32];

        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; ++i) {
            int n = n_base + i;
            if (n < N) {
                // Get Q8_0 block for weight row n, block k_block
                const block_q8_0* w_block = reinterpret_cast<const block_q8_0*>(
                    &weight_q8[n * num_blocks * 34 + k_block * 34]
                );

                float d_w = read_half_as_float(w_block->d);

                // Dot product
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    sum += act_block[j] * static_cast<float>(w_block->qs[j]);
                }
                acc[i] += sum * d_w;
            }
        }
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        int n = n_base + i;
        if (n < N) {
            output[m * N + n] = acc[i];
        }
    }
}

/**
 * Host wrapper - single kernel for all batch sizes
 */
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

    // Adjust N_PER_THREAD based on M for better performance
    int N_PER_THREAD = (M <= 8) ? 1 : 2;
    int threads_x = 16;
    int threads_y = 16;

    dim3 block(threads_x, threads_y);
    dim3 grid((N + threads_x * N_PER_THREAD - 1) / (threads_x * N_PER_THREAD),
              (M + threads_y - 1) / threads_y);

    if (N_PER_THREAD == 1) {
        q8_0_gemm_kernel<1><<<grid, block>>>(weight_q8, act, out, M, N, K);
    } else {
        q8_0_gemm_kernel<2><<<grid, block>>>(weight_q8, act, out, M, N, K);
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (W8A32C8 Q8_0)");
}
