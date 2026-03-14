#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

// Q4_0 block structure (packed)
struct block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // 32 packed 4-bit values
};

// Device function to read FP16 as float
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * Optimized W4A32C8 Q4_0 Quantized GEMM Kernel v4
 *
 * Key optimizations:
 * 1. Multi-thread-per-output for better utilization
 * 2. Shared memory for weight reuse across threads
 * 3. Vectorized memory access
 * 4. Optimized dot product computation
 */
template <int THREADS_PER_N, int BLOCKS_PER_THREAD>
__global__ void __launch_bounds__(256) w4a32c8_q4_0_gemm_kernel_v4(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each thread block processes one M row
    const int m = blockIdx.x;

    if (m >= M) return;

    const int num_blocks_k = K / 32;

    // Shared memory for caching weight blocks
    __shared__ float s_d_w[THREADS_PER_N * BLOCKS_PER_THREAD];
    __shared__ uint8_t s_qs[THREADS_PER_N * BLOCKS_PER_THREAD][16];

    // Each thread handles multiple N values
    const int n_base = blockIdx.y * (THREADS_PER_N * BLOCKS_PER_THREAD);
    const int n_start = threadIdx.y * BLOCKS_PER_THREAD + threadIdx.x;
    const int n_step = blockDim.y * blockDim.x;

    // Accumulators for each N this thread handles
    float acc[BLOCKS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < BLOCKS_PER_THREAD; i++) {
        acc[i] = 0.0f;
    }

    // Pre-compute activation pointer
    const float* a_row = activation + m * K;

    // Process K blocks
    for (int bk = 0; bk < num_blocks_k; bk++) {
        // Cooperatively load weight blocks into shared memory
        #pragma unroll
        for (int i = 0; i < BLOCKS_PER_THREAD; i++) {
            int load_idx = threadIdx.y * BLOCKS_PER_THREAD + threadIdx.x + i * blockDim.x;
            if (load_idx < THREADS_PER_N * BLOCKS_PER_THREAD) {
                int n = n_base + load_idx;
                if (n < N) {
                    const block_q4_0* w_block = reinterpret_cast<const block_q4_0*>(
                        weight + n * num_blocks_k * 18 + bk * 18
                    );
                    float d_w = read_half_as_float(w_block->d);
                    s_d_w[load_idx] = d_w;
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        s_qs[load_idx][j] = w_block->qs[j];
                    }
                }
            }
        }

        __syncthreads();

        // Each thread processes its assigned N values
        const float* a_block = a_row + bk * 32;

        // Dynamic quantization of activation block
        float a_max = 0.0f;
        float a_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            float a0 = a_block[i];
            float a1 = a_block[i + 1];
            float a2 = a_block[i + 2];
            float a3 = a_block[i + 3];

            a_max = fmaxf(a_max, fmaxf(fabsf(a0), fmaxf(fabsf(a1), fabsf(a2))));
            a_max = fmaxf(a_max, fabsf(a3));

            a_sum += a0 + a1 + a2 + a3;
        }

        const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

        // Pre-quantize activations to int8
        int8_t a_qs[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_qs[i] = __float2int_rn(a_block[i] / d_a);
        }

        // Compute dot products for assigned N values
        #pragma unroll
        for (int i = 0; i < BLOCKS_PER_THREAD; i++) {
            int n_local = threadIdx.y * BLOCKS_PER_THREAD + threadIdx.x + i * blockDim.x;
            if (n_local < THREADS_PER_N * BLOCKS_PER_THREAD) {
                int n = n_base + n_local;
                if (n < N) {
                    float d_w = s_d_w[n_local];

                    int32_t sumi = 0;
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint8_t byte_val = s_qs[n_local][j];
                        int w_low = byte_val & 0x0F;
                        int w_high = (byte_val >> 4) & 0x0F;

                        sumi += w_low * a_qs[j];
                        sumi += w_high * a_qs[j + 16];
                    }

                    acc[i] += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < BLOCKS_PER_THREAD; i++) {
        int n = n_base + threadIdx.y * BLOCKS_PER_THREAD + threadIdx.x + i * blockDim.x;
        if (n < N) {
            output[m * N + n] = acc[i];
        }
    }
}

/**
 * Host function to launch the optimized kernel
 */
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kByte, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    // Allocate output tensor
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Choose configuration based on M
    const int THREADS_PER_N = 32;   // Threads per N dimension
    const int BLOCKS_PER_THREAD = 2; // N values per thread

    const int n_blocks_total = (N + THREADS_PER_N * BLOCKS_PER_THREAD - 1) / (THREADS_PER_N * BLOCKS_PER_THREAD);

    dim3 grid(M, n_blocks_total);
    dim3 block(THREADS_PER_N, BLOCKS_PER_THREAD);

    // Launch optimized kernel
    w4a32c8_q4_0_gemm_kernel_v4<THREADS_PER_N, BLOCKS_PER_THREAD><<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v4 (DeepSeek-V2 LM Head)");
}
