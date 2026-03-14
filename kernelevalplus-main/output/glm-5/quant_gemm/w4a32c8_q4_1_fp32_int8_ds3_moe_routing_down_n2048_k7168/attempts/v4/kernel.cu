/**
 * Q4_1 x FP32 Quantized GEMM Kernel v4
 * Task: w4a32c8_q4_1_fp32_int8_ds3_moe_routing_down_n2048_k7168
 * Dimensions: N=2048, K=7168, M=variable
 *
 * Q4_1 format (20 bytes per block of 32 weights):
 *   - Bytes 0-1: FP16 scale (d_w)
 *   - Bytes 2-3: FP16 min (m_w)
 *   - Bytes 4-19: Packed 4-bit weights (16 bytes for 32 values)
 *
 * v4 Optimizations:
 * - Strategy dispatch based on M (small vs large batch)
 * - Small M (<=8): Use vectorized 2D tiling for memory efficiency
 * - Large M (>8): Use 1D grid with each block processing multiple rows
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32
#define TILE_N_SMALL 256  // Tile size for small M
#define TILE_N_LARGE 512  // Tile size for large M
#define TILE_M 8          // Rows processed per block for large M

// Device function to convert FP16 to FP32
__device__ __forceinline__ float fp16_to_fp32(uint16_t fp16_val) {
    __half h = *((__half*)&fp16_val);
    return __half2float(h);
}

// Device function to dequantize a Q4_1 block
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
 * Kernel for small M (memory-bound)
 * Each block processes one row and TILE_N_SMALL columns
 */
__global__ void q4_1_fp32_gemm_kernel_small(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int n_start = blockIdx.y * TILE_N_SMALL;
    const int num_blocks_k = K / BLOCK_SIZE;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;

    __shared__ float s_act[BLOCK_SIZE];

    const int outputs_per_thread = (TILE_N_SMALL + blockDimX - 1) / blockDimX;

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
 * Kernel for large M (compute-bound)
 * Each block processes TILE_M rows and TILE_N_LARGE columns
 */
__global__ void q4_1_fp32_gemm_kernel_large(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_start = blockIdx.x * TILE_M;
    const int n_start = blockIdx.y * TILE_N_LARGE;
    const int num_blocks_k = K / BLOCK_SIZE;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;

    __shared__ float s_act[BLOCK_SIZE];

    const int outputs_per_thread = (TILE_N_LARGE + blockDimX - 1) / blockDimX;

    // Accumulators for each row in the tile
    float acc[8 * TILE_M];  // Max 8 outputs * 8 rows
    #pragma unroll
    for (int i = 0; i < 8 * TILE_M; i++) {
        acc[i] = 0.0f;
    }

    // Process each row in the tile
    for (int m_offset = 0; m_offset < TILE_M; m_offset++) {
        int m_idx = m_start + m_offset;
        if (m_idx >= M) break;

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
                    acc[m_offset * 8 + out_idx] += sum;
                }
            }
            __syncthreads();
        }

        // Write results for this row
        float* out_row = output + m_idx * N;
        for (int out_idx = 0; out_idx < outputs_per_thread; out_idx++) {
            int n_idx = n_start + tid * outputs_per_thread + out_idx;
            if (n_idx < N) {
                out_row[n_idx] = acc[m_offset * 8 + out_idx];
            }
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

    if (M <= 8) {
        // Small batch: 2D grid, each block processes one row
        const int block_size = 256;
        const int num_tiles_n = (N + TILE_N_SMALL - 1) / TILE_N_SMALL;
        dim3 grid(M, num_tiles_n);

        q4_1_fp32_gemm_kernel_small<<<grid, block_size>>>(
            weight_ptr, act_ptr, out_ptr, M, N, K
        );
    } else {
        // Large batch: 2D grid, each block processes TILE_M rows
        const int block_size = 256;
        const int num_tiles_m = (M + TILE_M - 1) / TILE_M;
        const int num_tiles_n = (N + TILE_N_LARGE - 1) / TILE_N_LARGE;
        dim3 grid(num_tiles_m, num_tiles_n);

        q4_1_fp32_gemm_kernel_large<<<grid, block_size>>>(
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
