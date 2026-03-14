/**
 * Q4_1 x FP32 Quantized GEMM Kernel v6
 * Task: w4a32c8_q4_1_fp32_int8_ds3_moe_routing_down_n2048_k7168
 * Dimensions: N=2048, K=7168, M=variable
 *
 * v6 Strategy:
 * - Use v3's 2D tiling approach for all M values
 * - Optimize by reducing register pressure and improving instruction level parallelism
 * - Use larger block size for better utilization
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32
#define TILE_N 128  // Reduced tile size to reduce shared memory and improve occupancy

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
 * Unified kernel with optimized tile size
 * Each block processes TILE_N columns
 * Uses 128 threads per block (4 warps)
 */
__global__ void q4_1_fp32_gemm_kernel(
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

    float acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
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

    // Use 128 threads per block (4 warps)
    const int block_size = 128;
    const int num_tiles_n = (N + TILE_N - 1) / TILE_N;
    dim3 grid(M, num_tiles_n);

    q4_1_fp32_gemm_kernel<<<grid, block_size>>>(
        weight_ptr, act_ptr, out_ptr, M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 x FP32 Quantized GEMM");
}
