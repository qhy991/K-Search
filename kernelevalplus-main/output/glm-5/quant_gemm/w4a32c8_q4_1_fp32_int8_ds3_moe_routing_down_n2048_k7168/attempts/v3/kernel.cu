/**
 * Q4_1 x FP32 Quantized GEMM Kernel v3
 * Task: w4a32c8_q4_1_fp32_int8_ds3_moe_routing_down_n2048_k7168
 * Dimensions: N=2048, K=7168, M=variable
 *
 * Q4_1 format (20 bytes per block of 32 weights):
 *   - Bytes 0-1: FP16 scale (d_w)
 *   - Bytes 2-3: FP16 min (m_w)
 *   - Bytes 4-19: Packed 4-bit weights (16 bytes for 32 values)
 *
 * llama.cpp packing: low nibbles (positions 0-15) first, then high nibbles (positions 16-31)
 * Dequantization: w_fp32 = d_w * w_vals + m_w
 *
 * v3 Strategy:
 * - Fixed shared memory size limitation (48 KB per block on RTX 4090)
 * - Use shared memory for activation tiles only
 * - Dequantize weights on the fly with caching
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32
#define TILE_N 256  // Each block processes 256 columns
#define TILE_K 32   // Tile K dimension

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
    // Read FP16 scale (d_w) - bytes 0-1
    d_w = fp16_to_fp32(block_ptr[0] | (block_ptr[1] << 8));

    // Read FP16 min (m_w) - bytes 2-3
    m_w = fp16_to_fp32(block_ptr[2] | (block_ptr[3] << 8));

    // Unpack 4-bit weights (bytes 4-19, 16 bytes = 32 nibbles)
    const uint8_t* packed = block_ptr + 4;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out_values[i] = (float)(packed[i] & 0x0F);
        out_values[i + 16] = (float)((packed[i] >> 4) & 0x0F);
    }
}

/**
 * Optimized kernel with 2D tiling
 * Grid: [M, N/TILE_N]
 * Block: TILE_N threads
 */
__global__ void q4_1_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int n_start = n_tile * TILE_N;
    const int num_blocks_k = K / BLOCK_SIZE;

    const int tid = threadIdx.x;

    // Shared memory for activation tile
    __shared__ float s_act[TILE_K];

    // Each thread computes one output value (or multiple if TILE_N > blockDim.x)
    const int outputs_per_thread = (TILE_N + blockDim.x - 1) / blockDim.x;

    // Accumulators
    float acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i] = 0.0f;
    }

    // Pointer to this row's activation
    const float* act_row = activation + m_idx * K;

    // Iterate over K in blocks of 32
    for (int block_k = 0; block_k < num_blocks_k; block_k++) {
        // Load activation tile to shared memory (coalesced)
        if (tid < TILE_K) {
            s_act[tid] = act_row[block_k * TILE_K + tid];
        }
        __syncthreads();

        // Compute partial GEMM for this tile
        for (int out_idx = 0; out_idx < outputs_per_thread; out_idx++) {
            int n_idx = n_start + tid * outputs_per_thread + out_idx;
            if (n_idx < N) {
                const uint8_t* w_block_ptr = weight_q + n_idx * num_blocks_k * 20 + block_k * 20;

                // Dequantize weight block
                float w_vals[BLOCK_SIZE], d_w, m_w;
                dequantize_q4_1_block(w_block_ptr, w_vals, d_w, m_w);

                // Accumulate
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

    // Write results
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

    const int block_size = 256;  // TILE_N
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
