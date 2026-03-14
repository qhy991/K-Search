/**
 * W4A32C8 Q4_0 × FP32 GEMM for DeepSeek-V3 Attention Output - V3 Final
 *
 * Strategy: Split-K for small batch, tiled for large batch
 *
 * Computes: C = A @ W^T where:
 * - A is FP32 activation [M, K]
 * - W is Q4_0 quantized weight [N, K/32]
 *
 * Q4_0 block format (18 bytes):
 * - d: FP16 scale (2 bytes)
 * - qs: 16 bytes of packed 4-bit values
 *
 * Q8_1 dynamic quantization on activation:
 * - Find max absolute value in each 32-element block
 * - Scale: d_a = max_abs / 127
 * - Sum: s_a = sum(activation)
 * - Quantized: q_a[i] = round(activation[i] / d_a)
 *
 * Formula: result = d4_0 * (d8_1 * sumi - 8 * s8_1)
 * where sumi = dot(q_a, q_w) using DP4A
 *
 * Dimensions: M (variable), N = 7168, K = 7168
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK4_0 32
#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Q4_0 block structure (must match llama.cpp layout)
typedef struct {
    uint16_t d;          // FP16 scale/dequantization factor
    uint8_t qs[16];      // packed 4-bit values (32 values)
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size must be 18 bytes");

// Convert uint16 to float (FP16) - use union for better optimization
__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// DP4A intrinsic (available on CC >= 6.1)
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

/**
 * Process one Q4_0 block with Q8_1 dynamic quantization
 * Uses the formula: result = d_w * (d_a * sumi - 8 * s_a)
 */
__device__ __forceinline__ float process_q4_0_block(
    const block_q4_0* w_block,
    const float* act
) {
    // Extract weight scale
    const float d_w = half_to_float(w_block->d);

    // Q8_1 dynamic quantization: find scale and sum for activation
    float a_max = 0.0f;
    float s_a = 0.0f;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float4 v = *reinterpret_cast<const float4*>(&act[j * 4]);
        a_max = fmaxf(a_max, fabsf(v.x));
        a_max = fmaxf(a_max, fabsf(v.y));
        a_max = fmaxf(a_max, fabsf(v.z));
        a_max = fmaxf(a_max, fabsf(v.w));
        s_a += v.x + v.y + v.z + v.w;
    }

    const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
    const float inv_d_a = 1.0f / d_a;

    // Compute dot product using DP4A
    // Unpack Q4_0 in llama.cpp format: first all low nibbles, then all high nibbles
    int sumi = 0;
    const uint8_t* qs = w_block->qs;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t q0 = qs[j * 4 + 0];
        const uint8_t q1 = qs[j * 4 + 1];
        const uint8_t q2 = qs[j * 4 + 2];
        const uint8_t q3 = qs[j * 4 + 3];

        // Pack 4 Q4_0 low nibbles into 32-bit
        const int w_low = (q0 & 0x0F) | ((q1 & 0x0F) << 8) |
                         ((q2 & 0x0F) << 16) | ((q3 & 0x0F) << 24);

        // Pack 4 Q4_0 high nibbles into 32-bit
        const int w_high = (q0 >> 4) | ((q1 >> 4) << 8) |
                          ((q2 >> 4) << 16) | ((q3 >> 4) << 24);

        // Quantize activation (low nibbles: positions 0-15)
        const int a_low = ((int)(uint8_t)__float2int_rn(act[j * 4 + 0] * inv_d_a)) |
                         (((int)(uint8_t)__float2int_rn(act[j * 4 + 1] * inv_d_a)) << 8) |
                         (((int)(uint8_t)__float2int_rn(act[j * 4 + 2] * inv_d_a)) << 16) |
                         (((int)(uint8_t)__float2int_rn(act[j * 4 + 3] * inv_d_a)) << 24);

        // Quantize activation (high nibbles: positions 16-31)
        const int a_high = ((int)(uint8_t)__float2int_rn(act[j * 4 + 16] * inv_d_a)) |
                          (((int)(uint8_t)__float2int_rn(act[j * 4 + 17] * inv_d_a)) << 8) |
                          (((int)(uint8_t)__float2int_rn(act[j * 4 + 18] * inv_d_a)) << 16) |
                          (((int)(uint8_t)__float2int_rn(act[j * 4 + 19] * inv_d_a)) << 24);

        sumi = dp4a(a_low, w_low, sumi);
        sumi = dp4a(a_high, w_high, sumi);
    }

    // Apply the formula: d_w * (d_a * sumi - 8 * s_a)
    return d_w * (d_a * (float)sumi - 8.0f * s_a);
}

// ============================================================================
// Split-K Kernel for Small Batch (M <= 8)
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE)
gemm_split_k(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K,
    const int k_split) {

    __shared__ float partial_sums[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int block_n = blockIdx.x;
    const int split_id = blockIdx.y;
    const int row = blockIdx.z;

    if (block_n >= N || row >= M) return;

    const int num_k_blocks = K / QK4_0;
    const int k_start = split_id * k_split;
    const int k_end = min(k_start + k_split, num_k_blocks);

    float sum = 0.0f;

    // Each thread processes multiple blocks
    for (int kb = k_start + tid; kb < k_end; kb += BLOCK_SIZE) {
        const block_q4_0* w = &weight[block_n * num_k_blocks + kb];
        const float* act = &activation[row * K + kb * QK4_0];
        sum += process_q4_0_block(w, act);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        partial_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0) {
        sum = (lane_id < BLOCK_SIZE / WARP_SIZE) ? partial_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(&output[row * N + block_n], sum);
        }
    }
}

// ============================================================================
// Tiled Kernel for Large Batch (M > 8)
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE)
gemm_tiled(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {

    const int TILE_M = 4;
    const int TILE_N = 64;

    const int tid = threadIdx.x;
    const int row_start = blockIdx.y * TILE_M;
    const int col_start = blockIdx.x * TILE_N;

    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    const int local_row = warp_id / 2;
    const int local_col = lane_id + (warp_id % 2) * WARP_SIZE;

    const int row = row_start + local_row;
    const int col = col_start + local_col;

    const bool valid = (row < M) && (col < N);
    const int num_k_blocks = K / QK4_0;

    float sum = 0.0f;

    if (valid) {
        const block_q4_0* w_base = &weight[col * num_k_blocks];
        const float* a_base = &activation[row * K];

        for (int kb = 0; kb < num_k_blocks; ++kb) {
            sum += process_q4_0_block(&w_base[kb], &a_base[kb * QK4_0]);
        }
    }

    if (valid) {
        output[row * N + col] = sum;
    }
}

// ============================================================================
// PyTorch binding
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K) {

    // Use empty instead of zeros for better performance (output will be overwritten)
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const block_q4_0* weight_ptr = reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>());

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int num_k_blocks = K / QK4_0;

    if (M <= 8) {
        // Split-K for small batch - zero output first
        cudaMemset(output.data_ptr<float>(), 0, M * N * sizeof(float));

        int num_splits = max(1, (num_sms * 4) / (M * N));
        num_splits = min(num_splits, num_k_blocks);
        const int k_split = (num_k_blocks + num_splits - 1) / num_splits;

        dim3 grid(N, num_splits, M);
        dim3 block(BLOCK_SIZE);

        gemm_split_k<<<grid, block>>>(
            weight_ptr, activation.data_ptr<float>(), output.data_ptr<float>(),
            M, N, K, k_split);
    } else {
        // Tiled kernel for large batch
        const int TILE_M = 4;
        const int TILE_N = 64;

        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        dim3 block(BLOCK_SIZE);

        gemm_tiled<<<grid, block>>>(
            weight_ptr, activation.data_ptr<float>(), output.data_ptr<float>(),
            M, N, K);
    }

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 Quantized GEMM for DeepSeek-V3 Attention Output");
}
