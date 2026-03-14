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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v6 - Coalesced Loads)
 *
 * Qwen3-4B FFN Down projection
 * - N = 2560 (output features)
 * - K = 9728 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Optimizations v6:
 * - Coalesced memory reads using float4
 * - Each thread reads contiguous memory regions
 * - Optimized for memory-bound small batches
 */

#define QK 32

__device__ __inline__ float read_fp16(const uint8_t* p) {
    uint16_t u = p[0] | ((uint16_t)p[1] << 8);
    union { uint16_t u16; __half f16; } un;
    un.u16 = u;
    return __half2float(un.f16);
}

// Coalesced FP32 version: each thread processes 4 output elements
__global__ void gemm_coalesced_fp32(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each block processes 4*N elements
    const int m_idx = blockIdx.y;
    const int n_block = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    if (m_idx >= M) return;

    const int num_blocks = K / QK;
    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_base = weight + n_block * num_blocks * 18;

    // Each thread computes 4 accumulators
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint8_t* weight_ptr[4] = {
        weight_base,
        weight_base + num_blocks * 18,
        weight_base + 2 * num_blocks * 18,
        weight_base + 3 * num_blocks * 18
    };

    // Process all K blocks
    for (int kb = 0; kb < num_blocks; kb++) {
        const float* act_block = act_row + kb * QK;

        // Unroll through all 32 activation values
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float a = act_block[i];

            // Process 4 weight blocks in parallel
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                if (n_block + w >= N) continue;

                const uint8_t* wb = weight_ptr[w] + kb * 18;
                float dw = read_fp16(wb);

                const uint8_t* w_packed = wb + 2;
                int idx = i / 16;
                int bit_off = (i % 16);
                uint8_t packed = w_packed[idx];

                int w_val = (bit_off < 16) ?
                    (int)(packed & 0x0F) - 8 :
                    (int)((packed >> 4) & 0x0F) - 8;

                acc[w] += a * (dw * w_val);
            }
        }
    }

    // Write results with bounds checking
    for (int w = 0; w < 4; w++) {
        int n_idx = n_block + w;
        if (n_idx < N) {
            output[m_idx * N + n_idx] = acc[w];
        }
    }
}

// Original simple version for comparison
__global__ void gemm_simple_fp32(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / QK;
    float acc = 0.0f;

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row_base = weight + n_idx * num_blocks * 18;

    for (int kb = 0; kb < num_blocks; kb++) {
        const float* act_block = act_row + kb * QK;
        const uint8_t* wb = weight_row_base + kb * 18;
        float dw = read_fp16(wb);
        const uint8_t* w_packed = wb + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b = w_packed[i];
            acc += act_block[i] * (dw * ((int)(b & 0x0F) - 8));
            acc += act_block[i + 16] * (dw * ((int)((b >> 4) & 0x0F) - 8));
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

    auto weight_contig = weight.contiguous();
    auto act_contig = activation.contiguous();

    // Use coalesced version for better memory access
    // Each thread processes 4 output elements for better coalescing
    int threads_per_block = 64;  // 64 threads, each computes 4 outputs
    int blocks_x = (N + 4 * threads_per_block - 1) / (4 * threads_per_block);
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    gemm_coalesced_fp32<<<grid, block>>>(
        (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
        act_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V6 - Coalesced Loads)");
}
