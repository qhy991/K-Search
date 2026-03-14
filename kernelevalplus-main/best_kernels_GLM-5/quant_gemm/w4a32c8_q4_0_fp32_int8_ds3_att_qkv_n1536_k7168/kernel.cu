#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Block size for quantization
constexpr int BLOCK_SIZE = 32;

// Device function to convert FP16 (uint16) to FP32
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// Optimized dot product with max ILP
__device__ __forceinline__ float dot_q4_0_fp32(
    const uint8_t* __restrict__ w_block,
    const float* __restrict__ a
) {
    float d_w = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(w_block));
    const uint8_t* qs = w_block + 2;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t p0 = qs[i];
        uint8_t p1 = qs[i + 8];
        sum0 += (float)((int)(p0 & 0x0F) - 8) * a[i];
        sum1 += (float)((int)((p0 >> 4) & 0x0F) - 8) * a[i + 16];
        sum2 += (float)((int)(p1 & 0x0F) - 8) * a[i + 8];
        sum3 += (float)((int)((p1 >> 4) & 0x0F) - 8) * a[i + 24];
    }

    return d_w * (sum0 + sum1 + sum2 + sum3);
}

// Small M kernel with shared memory activation cache
// Focus on activation reuse across N dimension
constexpr int SM_TILE_N = 64;  // 64 columns per thread block
constexpr int SM_TILE_K = 32;  // 32 K-blocks at a time (1024 K values)

__global__ void __launch_bounds__(256) gemm_kernel_small_m_sm(
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int k_blocks = K / BLOCK_SIZE;
    const int m = blockIdx.y;
    if (m >= M) return;

    const int n_block = blockIdx.x;
    const int n_start = n_block * SM_TILE_N;
    const int tid = threadIdx.x;

    // Shared memory for activation (32 * 32 = 1024 floats = 4KB)
    __shared__ float a_shared[SM_TILE_K * BLOCK_SIZE];

    const float* __restrict__ a_row = activation + m * K;

    // Each thread handles multiple columns
    const int cols_per_thread = (SM_TILE_N + 255) / 256;

    float acc[8] = {0.0f};  // Up to 8 columns per thread

    // Process K in tiles
    for (int kb_tile = 0; kb_tile < k_blocks; kb_tile += SM_TILE_K) {
        const int kb_end = min(kb_tile + SM_TILE_K, k_blocks);
        const int kb_count = kb_end - kb_tile;

        // Cooperative load of activation tile
        for (int i = tid; i < kb_count * BLOCK_SIZE; i += 256) {
            int kb_local = i / BLOCK_SIZE;
            int k_local = i % BLOCK_SIZE;
            int global_kb = kb_tile + kb_local;
            a_shared[kb_local * BLOCK_SIZE + k_local] = a_row[global_kb * BLOCK_SIZE + k_local];
        }
        __syncthreads();

        // Compute for each column this thread handles
        for (int c = 0; c < cols_per_thread; ++c) {
            const int n = n_start + tid + c * 256;
            if (n >= N) break;

            for (int kb = 0; kb < kb_count; ++kb) {
                const uint8_t* w_block = weight_data + (n * k_blocks + kb_tile + kb) * 18;
                const float* a_vals = &a_shared[kb * BLOCK_SIZE];
                acc[c] += dot_q4_0_fp32(w_block, a_vals);
            }
        }
        __syncthreads();
    }

    // Write results
    for (int c = 0; c < cols_per_thread; ++c) {
        const int n = n_start + tid + c * 256;
        if (n >= N) break;
        output[m * N + n] = acc[c];
    }
}

// Simple 1D kernel for small M
__global__ void __launch_bounds__(256) gemm_kernel_1d(
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int k_blocks = K / BLOCK_SIZE;
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const float* __restrict__ a_row = activation + m * K;

    for (int n = tid; n < N; n += 256) {
        float acc = 0.0f;

        for (int kb = 0; kb < k_blocks; ++kb) {
            const int a_offset = kb * BLOCK_SIZE;
            float a_vals[BLOCK_SIZE];

            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i += 4) {
                float4 v = *reinterpret_cast<const float4*>(a_row + a_offset + i);
                a_vals[i+0] = v.x;
                a_vals[i+1] = v.y;
                a_vals[i+2] = v.z;
                a_vals[i+3] = v.w;
            }

            const uint8_t* w_block = weight_data + (n * k_blocks + kb) * 18;
            acc += dot_q4_0_fp32(w_block, a_vals);
        }

        output[m * N + n] = acc;
    }
}

// Large M kernel: 2D grid
__global__ void __launch_bounds__(256) gemm_kernel_2d(
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int k_blocks = K / BLOCK_SIZE;

    const int m = blockIdx.y * 16 + threadIdx.y;
    const int n = blockIdx.x * 16 + threadIdx.x;

    if (m >= M || n >= N) return;

    const float* __restrict__ a_row = activation + m * K;
    float acc = 0.0f;

    for (int kb = 0; kb < k_blocks; ++kb) {
        const int a_offset = kb * BLOCK_SIZE;
        float a_vals[BLOCK_SIZE];

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += 4) {
            float4 v = *reinterpret_cast<const float4*>(a_row + a_offset + i);
            a_vals[i+0] = v.x;
            a_vals[i+1] = v.y;
            a_vals[i+2] = v.z;
            a_vals[i+3] = v.w;
        }

        const uint8_t* w_block = weight_data + (n * k_blocks + kb) * 18;
        acc += dot_q4_0_fp32(w_block, a_vals);
    }

    output[m * N + n] = acc;
}

// PyTorch interface
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(activation.is_contiguous(), "Activation must be contiguous");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M <= 8) {
        // Small M: shared memory activation cache
        const int n_blocks = (N + SM_TILE_N - 1) / SM_TILE_N;
        const dim3 blocks(n_blocks, M);

        gemm_kernel_small_m_sm<<<blocks, 256>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 128) {
        // Medium M: 1D strided
        gemm_kernel_1d<<<M, 256>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large M: 2D grid
        const dim3 threads(16, 16);
        const dim3 blocks((N + 15) / 16, (M + 15) / 16);

        gemm_kernel_2d<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM v12");
}
