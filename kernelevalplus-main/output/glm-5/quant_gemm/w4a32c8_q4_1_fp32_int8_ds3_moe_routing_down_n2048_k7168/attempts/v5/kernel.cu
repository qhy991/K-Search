/**
 * Q4_1 x FP32 Quantized GEMM Kernel v5
 * Task: w4a32c8_q4_1_fp32_int8_ds3_moe_routing_down_n2048_k7168
 * Dimensions: N=2048, K=7168, M=variable
 *
 * v5 Optimizations:
 * - Based on v3 (best for small M)
 * - Improved large M by using larger tiles and better thread utilization
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32
#define TILE_N 512  // Each block processes 512 columns

__device__ __forceinline__ float fp16_to_fp32(uint16_t fp16_val) {
    __half h = *((__half*)&fp16_val);
    return __half2float(h);
}

__device__ __forceinline__ void dequantize_q4_1_block(
    const uint8_t* block_ptr,
    float* out_values,
    float& d_w,
    float& m_w
) {
    d_w = fp16_to_fp32(block_ptr[0] | (block_ptr[1] << 8));
    m_w = fp16_to_fp32(block_ptr[2] | (block_ptr[3] << 8));

    const uint8_t* packed = block_ptr + 4;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out_values[i] = (float)(packed[i] & 0x0F);
        out_values[i + 16] = (float)((packed[i] >> 4) & 0x0F);
    }
}

/**
 * Optimized kernel for small M (<= 8)
 */
__global__ void q4_1_fp32_gemm_kernel_small(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int n_start = blockIdx.y * TILE_N;
    const int num_blocks_k = K / BLOCK_SIZE;

    const int tid = threadIdx.x;

    __shared__ float s_act[BLOCK_SIZE];

    const int outputs_per_thread = (TILE_N + blockDim.x - 1) / blockDim.x;

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i] = 0.0f;
    }

    const float* act_row = activation + m_idx * K;

    for (int block_k = 0; block_k < num_blocks_k; block_k++) {
        if (tid < BLOCK_SIZE) {
            s_act[tid] = act_row[block_k * BLOCK_SIZE + tid];
        }
        __syncthreads();

        for (int out_idx = 0; out_idx < outputs_per_thread; out_idx++) {
            int n_idx = n_start + tid * outputs_per_thread + out_idx;
            if (n_idx < N) {
                const uint8_t* w_block_ptr = weight_q + n_idx * num_blocks_k * 20 + block_k * 20;

                float w_vals[BLOCK_SIZE], d_w, m_w;
                dequantize_q4_1_block(w_block_ptr, w_vals, d_w, m_w);

                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    float w_dequant = d_w * w_vals[i] + m_w;
                    sum += s_act[i] * w_dequant;
                }
                acc[out_idx] += sum;
            }
        }
        __syncthreads();
    }

    float* out_row = output + m_idx * N;
    for (int out_idx = 0; out_idx < outputs_per_thread; out_idx++) {
        int n_idx = n_start + tid * outputs_per_thread + out_idx;
        if (n_idx < N) {
            out_row[n_idx] = acc[out_idx];
        }
    }
}

/**
 * Optimized kernel for large M (> 8)
 * Each thread computes one output value directly
 * Better utilization for large batches
 */
__global__ void q4_1_fp32_gemm_kernel_large(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int num_blocks_k = K / BLOCK_SIZE;

    // Each thread computes one output value (or loops if total_threads < M*N)
    for (int out_flat = global_tid; out_flat < M * N; out_flat += total_threads) {
        int m_idx = out_flat / N;
        int n_idx = out_flat % N;

        float acc = 0.0f;
        const float* act_row = activation + m_idx * K;

        for (int block_k = 0; block_k < num_blocks_k; block_k++) {
            // Load activation values (vectorized)
            float act_block[BLOCK_SIZE];
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                act_block[i] = act_row[block_k * BLOCK_SIZE + i];
            }

            // Dequantize weight block
            const uint8_t* w_block_ptr = weight_q + n_idx * num_blocks_k * 20 + block_k * 20;
            float w_vals[BLOCK_SIZE], d_w, m_w;
            dequantize_q4_1_block(w_block_ptr, w_vals, d_w, m_w);

            // Accumulate
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                float w_dequant = d_w * w_vals[i] + m_w;
                acc += act_block[i] * w_dequant;
            }
        }

        output[m_idx * N + n_idx] = acc;
    }
}

torch::Tensor forward(
    torch::Tensor weight_q,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(weight_q.device()));

    const uint8_t* weight_ptr = weight_q.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    if (M <= 8) {
        // Small batch: use tiling with shared memory
        const int block_size = 256;
        const int num_tiles_n = (N + TILE_N - 1) / TILE_N;
        dim3 grid(M, num_tiles_n);

        q4_1_fp32_gemm_kernel_small<<<grid, block_size>>>(
            weight_ptr, act_ptr, out_ptr, M, N, K
        );
    } else {
        // Large batch: direct mapping, one thread per output value
        const int block_size = 256;
        const int total_outputs = M * N;
        const int num_blocks = (total_outputs + block_size - 1) / block_size;

        q4_1_fp32_gemm_kernel_large<<<num_blocks, block_size>>>(
            weight_ptr, act_ptr, out_ptr, M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 x FP32 Quantized GEMM");
}
