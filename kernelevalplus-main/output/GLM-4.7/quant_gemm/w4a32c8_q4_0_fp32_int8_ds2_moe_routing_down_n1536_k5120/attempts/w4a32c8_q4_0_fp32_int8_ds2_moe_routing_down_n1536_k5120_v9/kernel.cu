#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v9 - Best Combined)
 *
 * DeepSeek-V2 MoE Routing Down Projection
 * - N = 1536 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Optimizations v9:
 * - Combines best of v4 (2-block iteration) and v5 (shared memory)
 * - Single-block sync for minimal overhead
 * - Better thread block configuration
 */

#define Q4_0_BLOCK_SIZE 32

// Small M kernel: direct access, 2-block iteration
__global__ void quant_gemm_small_m_v9(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    float acc = 0.0f;

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_idx * num_blocks * 18;

    for (int kb = 0; kb + 1 < num_blocks; kb += 2) {
        const float* ab0 = act_row + kb * 32;
        const float* ab1 = ab0 + 32;
        const uint8_t* wb0 = weight_row + kb * 18;
        const uint8_t* wb1 = wb0 + 18;

        half d_w0, d_w1;
        memcpy(&d_w0, wb0, sizeof(half));
        memcpy(&d_w1, wb1, sizeof(half));
        float dw0 = __half2float(d_w0);
        float dw1 = __half2float(d_w1);

        const uint8_t* wqs0 = wb0 + 2;
        const uint8_t* wqs1 = wb1 + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b0 = wqs0[i];
            uint8_t b1 = wqs1[i];
            acc += ab0[i] * (dw0 * ((int)(b0 & 0x0F) - 8));
            acc += ab0[i + 16] * (dw0 * ((int)((b0 >> 4) & 0x0F) - 8));
            acc += ab1[i] * (dw1 * ((int)(b1 & 0x0F) - 8));
            acc += ab1[i + 16] * (dw1 * ((int)((b1 >> 4) & 0x0F) - 8));
        }
    }

    if (num_blocks % 2 == 1) {
        int kb = num_blocks - 1;
        const float* ab = act_row + kb * 32;
        const uint8_t* wb = weight_row + kb * 18;

        half d_w;
        memcpy(&d_w, wb, sizeof(half));
        float dw = __half2float(d_w);
        const uint8_t* wqs = wb + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = wqs[i];
            acc += ab[i] * (dw * ((int)(b & 0x0F) - 8));
            acc += ab[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
        }
    }

    output[m_idx * N + n_idx] = acc;
}

// Large M kernel: shared memory with 2-block processing
__global__ void quant_gemm_large_m_v9(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_base = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_base >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    float acc = 0.0f;

    __shared__ float act_shared[Q4_0_BLOCK_SIZE];

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_base * num_blocks * 18;

    // Process 2 blocks per iteration to reduce sync overhead
    int kb = 0;
    for (; kb + 1 < num_blocks; kb += 2) {
        // Load 2 activation blocks into shared memory
        if (threadIdx.x < Q4_0_BLOCK_SIZE) {
            act_shared[threadIdx.x] = act_row[kb * 32 + threadIdx.x];
            act_shared[32 + threadIdx.x] = act_row[(kb + 1) * 32 + threadIdx.x];
        }
        __syncthreads();

        // Process block 0
        {
            const uint8_t* wb = weight_row + kb * 18;
            half d_w;
            memcpy(&d_w, wb, sizeof(half));
            float dw = __half2float(d_w);
            const uint8_t* wqs = wb + 2;
            const float* as = act_shared;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t b = wqs[i];
                acc += as[i] * (dw * ((int)(b & 0x0F) - 8));
                acc += as[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
            }
        }

        // Process block 1
        {
            const uint8_t* wb = weight_row + (kb + 1) * 18;
            half d_w;
            memcpy(&d_w, wb, sizeof(half));
            float dw = __half2float(d_w);
            const uint8_t* wqs = wb + 2;
            const float* as = act_shared + 32;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t b = wqs[i];
                acc += as[i] * (dw * ((int)(b & 0x0F) - 8));
                acc += as[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
            }
        }

        __syncthreads();
    }

    // Handle remaining block
    if (num_blocks % 2 == 1) {
        int kb = num_blocks - 1;
        if (threadIdx.x < Q4_0_BLOCK_SIZE) {
            act_shared[threadIdx.x] = act_row[kb * 32 + threadIdx.x];
        }
        __syncthreads();

        const uint8_t* wb = weight_row + kb * 18;
        half d_w;
        memcpy(&d_w, wb, sizeof(half));
        float dw = __half2float(d_w);
        const uint8_t* wqs = wb + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = wqs[i];
            acc += act_shared[i] * (dw * ((int)(b & 0x0F) - 8));
            acc += act_shared[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
        }
    }

    output[m_idx * N + n_base] = acc;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    auto weight_contig = weight.contiguous();
    auto act_contig = activation.contiguous();

    int threads_per_block = 256;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    // Strategy dispatch
    if (M < 8) {
        quant_gemm_small_m_v9<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        quant_gemm_large_m_v9<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V9 Best Combined)");
}
