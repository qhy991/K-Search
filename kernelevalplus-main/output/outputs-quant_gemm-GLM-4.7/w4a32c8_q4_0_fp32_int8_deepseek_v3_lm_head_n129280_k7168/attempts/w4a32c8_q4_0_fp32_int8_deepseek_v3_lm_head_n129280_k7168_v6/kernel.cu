#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// Q4_0 block structure
struct block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // 32 packed 4-bit values
};

__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * W4A32C8 Q4_0 Quantized GEMM Kernel v6 - Shared memory optimized
 *
 * Key optimizations:
 * 1. Cache weight tiles in shared memory
 * 2. Process multiple N values per thread block
 * 3. Each thread processes multiple outputs
 */
template<int TILE_N, int BLOCK_SIZE>
__global__ void w4a32c8_q4_0_gemm_kernel_v6(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Shared memory for weight tiles
    __shared__ float s_weight_scale[TILE_N];  // Scale for each N in tile
    __shared__ uint8_t s_weight_qs[TILE_N * 16];  // Packed values for each N in tile

    const int m_block = blockIdx.y;
    const int n_tile = blockIdx.x;
    const int tid = threadIdx.x;

    // Each thread block computes TILE_N output elements
    const int n_base = n_tile * TILE_N;
    const int n_local = tid;

    if (m_block >= M || n_base >= N) return;

    const int num_blocks = K / 32;

    // Initialize accumulators
    float acc[TILE_N / BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < TILE_N / BLOCK_SIZE; i++) {
        acc[i] = 0.0f;
    }

    // Process all blocks in K dimension
    for (int b = 0; b < num_blocks; b++) {
        // Load weight tile into shared memory
        #pragma unroll
        for (int i = 0; i < TILE_N; i += BLOCK_SIZE) {
            int n = n_base + i + n_local;
            if (n < N) {
                const block_q4_0* w_block = reinterpret_cast<const block_q4_0*>(
                    weight + n * num_blocks * 18 + b * 18
                );
                if (n_local < i + (TILE_N / BLOCK_SIZE)) {
                    s_weight_scale[i + n_local - i] = read_half_as_float(w_block->d);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        s_weight_qs[(i + n_local - i) * 16 + j] = w_block->qs[j];
                    }
                }
            }
        }
        __syncthreads();

        // Load activation block
        const float* a_block = activation + m_block * K + b * 32;

        // Find activation scale
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
        }
        float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

        // Compute activation sum
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_sum += a_block[i];
        }

        // Quantize activations
        int8_t a_quantized[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_quantized[i] = __float2int_rn(a_block[i] / d_a);
        }

        // Compute dot products for each N in tile
        #pragma unroll
        for (int i = 0; i < TILE_N / BLOCK_SIZE; i++) {
            int n = n_local + i * BLOCK_SIZE;
            if (n < TILE_N && (n_base + n) < N) {
                float d_w = s_weight_scale[n];
                int32_t sumi = 0;

                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed = s_weight_qs[n * 16 + j];
                    int w_low = packed & 0x0F;
                    int w_high = (packed >> 4) & 0x0F;
                    sumi += w_low * a_quantized[j];
                    sumi += w_high * a_quantized[j + 16];
                }

                acc[i] += d_w * (d_a * (float)sumi - 8.0f * a_sum);
            }
        }
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < TILE_N / BLOCK_SIZE; i++) {
        int n = n_base + n_local + i * BLOCK_SIZE;
        if (n < N) {
            output[m_block * N + n] = acc[i];
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
    TORCH_CHECK(weight.dtype() == torch::kByte, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    constexpr int TILE_N = 64;
    constexpr int BLOCK_SIZE = 64;

    const int blocks_x = (N + TILE_N - 1) / TILE_N;
    const int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(BLOCK_SIZE);

    w4a32c8_q4_0_gemm_kernel_v6<TILE_N, BLOCK_SIZE><<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v6 (DeepSeek-V3 LM Head)");
}
