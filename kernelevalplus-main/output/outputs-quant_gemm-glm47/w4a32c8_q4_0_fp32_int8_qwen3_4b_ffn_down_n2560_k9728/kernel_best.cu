/**
 * W4A32C8 Q4_0 Quantized GEMM Kernel - Combined Strategy
 * For Qwen3-4B FFN Down projection: M×9728 @ (9728/32)×2560 -> M×2560
 *
 * Q4_0 format (llama.cpp compatible): 18 bytes per block of 32 values
 *   - d: FP16 scale (2 bytes)
 *   - qs: uint8[16] (16 bytes) - 32 packed 4-bit values
 *   - Packing: byte[i] = q[i] (low nibble, positions 0-15) | (q[i+16] << 4) (high nibble, positions 16-31)
 *   - Quantization: q = round(val / d + 8), val = d * (q - 8)
 *
 * Strategy Dispatch:
 * - Small batch (M <= 8): Memory-optimized kernel from v1
 * - Large batch (M > 8): Compute-optimized kernel from v1
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * Q4_0 block dot product
 */
__device__ __forceinline__ float q4_0_block_dot(
    const uint8_t* __restrict__ w_block,
    const float* __restrict__ act_vals
) {
    // Read scale (first 2 bytes as FP16)
    half scale_half;
    memcpy(&scale_half, w_block, 2);
    float scale = __half2float(scale_half);

    const uint8_t* qs = w_block + 2;  // Packed 4-bit values (16 bytes)

    float acc = 0.0f;

    // Unroll for better performance
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t packed = qs[i];
        int q_low = packed & 0x0F;
        int q_high = (packed >> 4) & 0x0F;

        float w_low = scale * (float)(q_low - 8);
        float w_high = scale * (float)(q_high - 8);

        acc += w_low * act_vals[i];
        acc += w_high * act_vals[i + 16];
    }

    return acc;
}

/**
 * Small-batch kernel (M <= 8)
 * Memory-optimized: Each block handles one row, threads handle columns
 */
__global__ void gemm_w4a32c8_q4_0_small_batch_kernel(
    const uint8_t* __restrict__ weight_q,  // [N, K/32, 18]
    const float* __restrict__ activation,   // [M, K]
    float* __restrict__ output,           // [M, N]
    int M, int N, int K
) {
    const int m = blockIdx.x;
    const int BLOCK_N = blockDim.x;
    const int n_base = blockIdx.y * BLOCK_N;
    const int tid = threadIdx.x;
    const int n = n_base + tid;

    if (m >= M || n >= N) return;

    const int K_blocks = K / 32;
    const float* act_row = activation + m * K;

    // Compute output[m, n] = dot(activation[m,:], weight[n,:])
    float acc = 0.0f;

    for (int kb = 0; kb < K_blocks; kb++) {
        // Pointer to weight block for column n, block kb
        const uint8_t* w_block = weight_q + (n * K_blocks + kb) * 18;

        // Activation values for this block
        const float* act_vals = act_row + kb * 32;

        acc += q4_0_block_dot(w_block, act_vals);
    }

    output[m * N + n] = acc;
}

/**
 * Large-batch kernel (M > 8)
 * Compute-optimized: 2D tiling
 */
template<int TILE_M, int TILE_N>
__global__ void gemm_w4a32c8_q4_0_large_batch_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    // Thread position within block
    const int tid_m = threadIdx.y;
    const int tid_n = threadIdx.x;
    const int m = block_m + tid_m;
    const int n = block_n + tid_n;

    const int K_blocks = K / 32;

    // Each thread computes one output element output[m, n]
    float acc = 0.0f;

    // Iterate over K dimension in blocks
    for (int kb = 0; kb < K_blocks; kb++) {
        if (m < M && n < N) {
            const uint8_t* w_block = weight_q + (n * K_blocks + kb) * 18;
            const float* act_vals = activation + m * K + kb * 32;
            acc += q4_0_block_dot(w_block, act_vals);
        }
    }

    // Write result
    if (m < M && n < N) {
        output[m * N + n] = acc;
    }
}

/**
 * Host function with strategy dispatch
 * Automatically selects optimal kernel based on batch size M
 */
torch::Tensor forward(
    torch::Tensor weight_q,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight_q.device()));

    const uint8_t* d_wq = weight_q.data_ptr<uint8_t>();
    const float* d_act = activation.data_ptr<float>();
    float* d_out = output.data_ptr<float>();

    const int K_blocks = K / 32;

    // Strategy dispatch based on batch size
    if (M <= 8) {
        // Small batch: Use memory-optimized kernel (same as v1)
        // Each block handles one row, threads in block handle columns
        const int BLOCK_N = 256;
        dim3 block(BLOCK_N, 1, 1);
        dim3 grid(M, (N + BLOCK_N - 1) / BLOCK_N, 1);

        gemm_w4a32c8_q4_0_small_batch_kernel<<<grid, block>>>(
            d_wq, d_act, d_out, M, N, K
        );
    } else {
        // Large batch: Use compute-optimized tiled kernel (same as v1)
        // Use 32x32 tiles for good balance
        const int TILE_M = 32;
        const int TILE_N = 32;

        dim3 block(TILE_N, TILE_M, 1);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

        gemm_w4a32c8_q4_0_large_batch_kernel<TILE_M, TILE_N><<<grid, block>>>(
            d_wq, d_act, d_out, M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM Combined Strategy");
}
