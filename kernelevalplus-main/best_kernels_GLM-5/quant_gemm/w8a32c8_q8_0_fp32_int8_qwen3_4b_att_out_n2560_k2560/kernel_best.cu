/**
 * Quantized GEMM for Qwen3-4B Attention Output Projection (v4)
 *
 * Parameters: N = 2560, K = 2560, M = batch size
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 *
 * v4: Match final kernel approach with tuning
 * - No shared memory (rely on L1/L2 cache)
 * - Direct FP32 x INT8 multiplication
 * - Tune thread/block configuration
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK8_0 32
#define WARP_SIZE 32
#define Q8_0_BLOCK 34

inline __device__ float read_half_as_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

/**
 * M=1 kernel - match final config: 64 threads, 4 cols/block
 * This achieves 89.5% of baseline
 */
__global__ void __launch_bounds__(64)
gemm_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_k_blocks = K / QK8_0;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // 4 columns per block, 2 columns per warp
    const int base_col = blockIdx.x * 4 + warp_id * 2;
    const int cols_to_process = min(2, N - base_col);

    if (base_col >= N) return;

    float sums[2] = {0.0f, 0.0f};

    // K-parallel: each lane processes different K blocks
    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const float* act_ptr = activation + kb * QK8_0;

        // Vectorized load of 32 floats
        float4 a0 = *reinterpret_cast<const float4*>(act_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(act_ptr + 4);
        float4 a2 = *reinterpret_cast<const float4*>(act_ptr + 8);
        float4 a3 = *reinterpret_cast<const float4*>(act_ptr + 12);
        float4 a4 = *reinterpret_cast<const float4*>(act_ptr + 16);
        float4 a5 = *reinterpret_cast<const float4*>(act_ptr + 20);
        float4 a6 = *reinterpret_cast<const float4*>(act_ptr + 24);
        float4 a7 = *reinterpret_cast<const float4*>(act_ptr + 28);

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q8_0_BLOCK;
            const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
            const int8_t* qs = reinterpret_cast<const int8_t*>(w_block + 2);

            float block_sum = 0.0f;
            block_sum += a0.x * (float)qs[0] + a0.y * (float)qs[1] + a0.z * (float)qs[2] + a0.w * (float)qs[3];
            block_sum += a1.x * (float)qs[4] + a1.y * (float)qs[5] + a1.z * (float)qs[6] + a1.w * (float)qs[7];
            block_sum += a2.x * (float)qs[8] + a2.y * (float)qs[9] + a2.z * (float)qs[10] + a2.w * (float)qs[11];
            block_sum += a3.x * (float)qs[12] + a3.y * (float)qs[13] + a3.z * (float)qs[14] + a3.w * (float)qs[15];
            block_sum += a4.x * (float)qs[16] + a4.y * (float)qs[17] + a4.z * (float)qs[18] + a4.w * (float)qs[19];
            block_sum += a5.x * (float)qs[20] + a5.y * (float)qs[21] + a5.z * (float)qs[22] + a5.w * (float)qs[23];
            block_sum += a6.x * (float)qs[24] + a6.y * (float)qs[25] + a6.z * (float)qs[26] + a6.w * (float)qs[27];
            block_sum += a7.x * (float)qs[28] + a7.y * (float)qs[29] + a7.z * (float)qs[30] + a7.w * (float)qs[31];

            sums[c] += d_w * block_sum;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[base_col + c] = sums[c];
        }
    }
}

/**
 * Small M kernel (M <= 8): 128 threads, 16 columns per block
 */
__global__ void __launch_bounds__(128)
gemm_small_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.y;
    const int col_block_idx = blockIdx.x;

    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int COLS_PER_WARP = 4;
    const int base_col = col_block_idx * (num_warps * COLS_PER_WARP) + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK8_0;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const float* act_ptr = &activation[static_cast<int64_t>(row) * K + kb * QK8_0];

        float4 a0 = *reinterpret_cast<const float4*>(act_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(act_ptr + 4);
        float4 a2 = *reinterpret_cast<const float4*>(act_ptr + 8);
        float4 a3 = *reinterpret_cast<const float4*>(act_ptr + 12);
        float4 a4 = *reinterpret_cast<const float4*>(act_ptr + 16);
        float4 a5 = *reinterpret_cast<const float4*>(act_ptr + 20);
        float4 a6 = *reinterpret_cast<const float4*>(act_ptr + 24);
        float4 a7 = *reinterpret_cast<const float4*>(act_ptr + 28);

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q8_0_BLOCK;
            const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
            const int8_t* qs = reinterpret_cast<const int8_t*>(w_block + 2);

            float block_sum = 0.0f;
            block_sum += a0.x * (float)qs[0] + a0.y * (float)qs[1] + a0.z * (float)qs[2] + a0.w * (float)qs[3];
            block_sum += a1.x * (float)qs[4] + a1.y * (float)qs[5] + a1.z * (float)qs[6] + a1.w * (float)qs[7];
            block_sum += a2.x * (float)qs[8] + a2.y * (float)qs[9] + a2.z * (float)qs[10] + a2.w * (float)qs[11];
            block_sum += a3.x * (float)qs[12] + a3.y * (float)qs[13] + a3.z * (float)qs[14] + a3.w * (float)qs[15];
            block_sum += a4.x * (float)qs[16] + a4.y * (float)qs[17] + a4.z * (float)qs[18] + a4.w * (float)qs[19];
            block_sum += a5.x * (float)qs[20] + a5.y * (float)qs[21] + a5.z * (float)qs[22] + a5.w * (float)qs[23];
            block_sum += a6.x * (float)qs[24] + a6.y * (float)qs[25] + a6.z * (float)qs[26] + a6.w * (float)qs[27];
            block_sum += a7.x * (float)qs[28] + a7.y * (float)qs[29] + a7.z * (float)qs[30] + a7.w * (float)qs[31];

            sums[c] += d_w * block_sum;
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[static_cast<int64_t>(row) * N + base_col + c] = sums[c];
        }
    }
}

/**
 * Large M kernel: 256 threads, 64 columns per block
 */
__global__ void __launch_bounds__(256)
gemm_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int COLS_PER_WARP = 8;
    const int base_col = blockIdx.y * 8 * COLS_PER_WARP + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK8_0;
    float sums[8] = {0.0f};

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const float* act_ptr = &activation[static_cast<int64_t>(row) * K + kb * QK8_0];

        float4 a0 = *reinterpret_cast<const float4*>(act_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(act_ptr + 4);
        float4 a2 = *reinterpret_cast<const float4*>(act_ptr + 8);
        float4 a3 = *reinterpret_cast<const float4*>(act_ptr + 12);
        float4 a4 = *reinterpret_cast<const float4*>(act_ptr + 16);
        float4 a5 = *reinterpret_cast<const float4*>(act_ptr + 20);
        float4 a6 = *reinterpret_cast<const float4*>(act_ptr + 24);
        float4 a7 = *reinterpret_cast<const float4*>(act_ptr + 28);

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q8_0_BLOCK;
            const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
            const int8_t* qs = reinterpret_cast<const int8_t*>(w_block + 2);

            float block_sum = 0.0f;
            block_sum += a0.x * (float)qs[0] + a0.y * (float)qs[1] + a0.z * (float)qs[2] + a0.w * (float)qs[3];
            block_sum += a1.x * (float)qs[4] + a1.y * (float)qs[5] + a1.z * (float)qs[6] + a1.w * (float)qs[7];
            block_sum += a2.x * (float)qs[8] + a2.y * (float)qs[9] + a2.z * (float)qs[10] + a2.w * (float)qs[11];
            block_sum += a3.x * (float)qs[12] + a3.y * (float)qs[13] + a3.z * (float)qs[14] + a3.w * (float)qs[15];
            block_sum += a4.x * (float)qs[16] + a4.y * (float)qs[17] + a4.z * (float)qs[18] + a4.w * (float)qs[19];
            block_sum += a5.x * (float)qs[20] + a5.y * (float)qs[21] + a5.z * (float)qs[22] + a5.w * (float)qs[23];
            block_sum += a6.x * (float)qs[24] + a6.y * (float)qs[25] + a6.z * (float)qs[26] + a6.w * (float)qs[27];
            block_sum += a7.x * (float)qs[28] + a7.y * (float)qs[29] + a7.z * (float)qs[30] + a7.w * (float)qs[31];

            sums[c] += d_w * block_sum;
        }
    }

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[static_cast<int64_t>(row) * N + base_col + c] = sums[c];
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(activation.is_contiguous(), "Activation must be contiguous");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    const float* a_ptr = activation.data_ptr<float>();
    float* c_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(weight.device().index());

    if (M == 1) {
        // M=1: 64 threads (2 warps), 4 cols/block = 640 blocks for N=2560
        dim3 grid((N + 3) / 4, 1);
        dim3 block(64);
        gemm_m1_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    } else if (M <= 8) {
        // Small M: 128 threads, 16 cols/block
        const int n_blocks = (N + 15) / 16;
        dim3 grid(n_blocks, M);
        dim3 block(128);
        gemm_small_m_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    } else {
        // M>8: 256 threads, 64 cols/block
        dim3 grid(M, (N + 63) / 64);
        dim3 block(256);
        gemm_large_m_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W8A32C8 Q8_0 Quantized GEMM v4");
}
