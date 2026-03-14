/**
 * Q4_1 x FP32 Quantized GEMM Kernel v2
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
 * v2 Optimizations:
 * - Shared memory for weight blocks to avoid redundant dequantization
 * - Better thread mapping for coalesced memory access
 * - Vectorized loads for activations
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32
#define TILE_K 32
#define TILE_N 64  // Each block processes TILE_N columns

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
    // llama.cpp packing: low nibbles first (positions 0-15), then high nibbles (16-31)
    const uint8_t* packed = block_ptr + 4;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out_values[i] = (float)(packed[i] & 0x0F);       // positions 0-15 (low nibbles)
        out_values[i + 16] = (float)((packed[i] >> 4) & 0x0F);  // positions 16-31 (high nibbles)
    }
}

/**
 * Optimized kernel for small M (memory-bound)
 * Each thread block processes one row of M and TILE_N columns
 * Threads collaborate to reduce dequantization overhead
 */
__global__ void q4_1_fp32_gemm_kernel_small(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    const int num_blocks_k = K / BLOCK_SIZE;
    const int num_blocks_n = (N + TILE_N - 1) / TILE_N;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory for weight block dequantization
    __shared__ float s_w_vals[TILE_N][BLOCK_SIZE];
    __shared__ float s_d_w[TILE_N];
    __shared__ float s_m_w[TILE_N];

    // Each thread computes N_PER_THREAD output values
    const int N_PER_THREAD = TILE_N / blockDim.x;
    const int n_start = blockIdx.y * TILE_N;
    const int n_thread_start = tid * N_PER_THREAD;

    // Accumulators
    float acc[2];  // Up to 2 values per thread
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        acc[i] = 0.0f;
    }

    // Pointer to this row's activation
    const float* act_row = activation + m_idx * K;

    // Iterate over K in blocks of 32
    for (int block_k = 0; block_k < num_blocks_k; block_k++) {
        // Load activation block (coalesced - all threads read the same values)
        float act_block[BLOCK_SIZE];
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            act_block[i] = act_row[block_k * BLOCK_SIZE + i];
        }

        // Load weight blocks for TILE_N columns
        // Each thread dequantizes one weight block
        if (tid < TILE_N) {
            int n_idx = n_start + tid;
            if (n_idx < N) {
                const uint8_t* w_block_ptr = weight_q + n_idx * num_blocks_k * 20 + block_k * 20;
                dequantize_q4_1_block(w_block_ptr, s_w_vals[tid], s_d_w[tid], s_m_w[tid]);

                // Apply dequantization: w_fp32 = d_w * w_vals + m_w
                #pragma unroll
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    s_w_vals[tid][i] = s_d_w[tid] * s_w_vals[tid][i] + s_m_w[tid];
                }
            }
        }
        __syncthreads();

        // Compute partial GEMM for this thread's assigned columns
        for (int i = 0; i < N_PER_THREAD && (n_thread_start + i) < TILE_N; i++) {
            int w_idx = n_thread_start + i;
            if (n_start + w_idx < N) {
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    sum += act_block[j] * s_w_vals[w_idx][j];
                }
                acc[i] += sum;
            }
        }
        __syncthreads();
    }

    // Write results
    float* out_row = output + m_idx * N;
    for (int i = 0; i < N_PER_THREAD; i++) {
        int n = n_start + n_thread_start + i;
        if (n < N) {
            out_row[n] = acc[i];
        }
    }
}

/**
 * Optimized kernel for large M (compute-bound)
 * Uses 2D tiling and efficient memory access
 */
__global__ void q4_1_fp32_gemm_kernel_large(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_SIZE;

    extern __shared__ float s_act[];  // Shared memory for activation tile
    __shared__ float s_w_vals[TILE_N][BLOCK_SIZE];
    __shared__ float s_d_w[TILE_N];
    __shared__ float s_m_w[TILE_N];

    const int m_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_start = blockIdx.y * TILE_N;
    const int N_PER_THREAD = TILE_N / blockDim.x;
    const int n_thread_start = tid * N_PER_THREAD;

    // Accumulators
    float acc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        acc[i] = 0.0f;
    }

    // Pointer to this row's activation
    const float* act_row = activation + m_idx * K;

    // Iterate over K in blocks of 32
    for (int block_k = 0; block_k < num_blocks_k; block_k++) {
        // Load activation block to shared memory
        if (tid < BLOCK_SIZE) {
            s_act[tid] = act_row[block_k * BLOCK_SIZE + tid];
        }
        __syncthreads();

        // Load and dequantize weight blocks
        if (tid < TILE_N) {
            int n_idx = n_start + tid;
            if (n_idx < N) {
                const uint8_t* w_block_ptr = weight_q + n_idx * num_blocks_k * 20 + block_k * 20;
                dequantize_q4_1_block(w_block_ptr, s_w_vals[tid], s_d_w[tid], s_m_w[tid]);

                // Apply dequantization
                #pragma unroll
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    s_w_vals[tid][i] = s_d_w[tid] * s_w_vals[tid][i] + s_m_w[tid];
                }
            }
        }
        __syncthreads();

        // Compute partial GEMM
        for (int i = 0; i < N_PER_THREAD; i++) {
            int w_idx = n_thread_start + i;
            if (n_start + w_idx < N) {
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    sum += s_act[j] * s_w_vals[w_idx][j];
                }
                acc[i] += sum;
            }
        }
        __syncthreads();
    }

    // Write results
    float* out_row = output + m_idx * N;
    for (int i = 0; i < N_PER_THREAD; i++) {
        int n = n_start + n_thread_start + i;
        if (n < N) {
            out_row[n] = acc[i];
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

    // Choose kernel based on M
    if (M <= 8) {
        // Memory-bound small batch
        const int block_size = 32;  // One warp per block
        const int num_blocks_n = (N + TILE_N - 1) / TILE_N;
        dim3 grid(M, num_blocks_n);

        size_t shared_mem = TILE_N * BLOCK_SIZE * sizeof(float) +
                          TILE_N * sizeof(float) * 2;  // s_w_vals + s_d_w + s_m_w

        q4_1_fp32_gemm_kernel_small<<<grid, block_size, shared_mem>>>(
            weight_ptr, act_ptr, out_ptr, M, N, K
        );
    } else {
        // Compute-bound larger batch
        const int block_size = 32;
        const int num_blocks_n = (N + TILE_N - 1) / TILE_N;
        dim3 grid(M, num_blocks_n);

        size_t shared_mem = BLOCK_SIZE * sizeof(float) +  // s_act
                          TILE_N * BLOCK_SIZE * sizeof(float) +
                          TILE_N * sizeof(float) * 2;  // s_w_vals + s_d_w + s_m_w

        q4_1_fp32_gemm_kernel_large<<<grid, block_size, shared_mem>>>(
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
