/**
 * Final Optimized Q4_1 Quantized GEMM for Qwen3-4B FFN Down
 *
 * Dimensions: M (var), N=2560, K=9728
 * Weight format: Q4_1 (20 bytes per 32 values: 2B scale + 2B min + 16B packed 4-bit)
 * Formula: result = d_w * d_a * sumi + m_w * s_a
 *
 * RTX 4090 Performance target:
 *   - M=1: 1000+ GFLOPS (warp kernel with SM saturation)
 *   - M=512: 4000+ GFLOPS (column reuse with activation caching)
 *
 * Strategy:
 *   - M <= 16: Warp kernel (one output per warp, K distributed across lanes)
 *   - M > 16:  Column-reuse kernel (each lane processes K for multiple columns)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK4_1 32
#define WARP_SIZE 32

struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

inline __device__ float read_half_as_float(uint16_t h) {
    return __half2float(reinterpret_cast<half&>(h));
}

inline __device__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Strategy 1: Warp kernel for M <= 16
// Each warp computes ONE output element, K blocks distributed across lanes
// ============================================================================

__global__ void __launch_bounds__(256)
gemm_warp_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    const int num_k_blocks = K / QK4_1;

    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int row = idx / N;
        const int col = idx % N;

        float sum = 0.0f;
        const float* act_row = &activation[static_cast<int64_t>(row) * K];
        const block_q4_1* w_col = &weight[static_cast<int64_t>(col) * num_k_blocks];

        for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
            const block_q4_1* w_block = &w_col[kb];
            const float d_w = read_half_as_float(w_block->d);
            const float m_w = read_half_as_float(w_block->m);
            const int k_start = kb * QK4_1;

            float a_block[32];
            const float* act_ptr = &act_row[k_start];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
                a_block[i * 4 + 0] = a4.x;
                a_block[i * 4 + 1] = a4.y;
                a_block[i * 4 + 2] = a4.z;
                a_block[i * 4 + 3] = a4.w;
            }

            float a_max = 0.0f, a_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                a_max = fmaxf(a_max, fabsf(a_block[i]));
                a_sum += a_block[i];
            }
            const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

            int32_t a_packed[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
                a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                              ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
            }

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[i * 4]);
                int8_t w0 = (int8_t)(w_packed & 0x0F);
                int8_t w1 = (int8_t)((w_packed >> 8) & 0x0F);
                int8_t w2 = (int8_t)((w_packed >> 16) & 0x0F);
                int8_t w3 = (int8_t)((w_packed >> 24) & 0x0F);
                int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                            ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);

                w0 = (int8_t)((w_packed >> 4) & 0x0F);
                w1 = (int8_t)((w_packed >> 12) & 0x0F);
                w2 = (int8_t)((w_packed >> 20) & 0x0F);
                w3 = (int8_t)((w_packed >> 28) & 0x0F);
                w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                        ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }
            sum += d_w * d_a * (float)sumi + m_w * a_sum;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) output[row * N + col] = sum;
    }
}

// ============================================================================
// Strategy 2: Column-reuse kernel for M > 16
// Each lane processes K for multiple columns, reusing quantized activation
// ============================================================================

constexpr int COLS_PER_WARP = 16;

__global__ void __launch_bounds__(512)
gemm_column_reuse_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    const int col_block = blockIdx.y;
    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    constexpr int WARPS_PER_BLOCK = 512 / WARP_SIZE;  // 16
    constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * COLS_PER_WARP;  // 256

    const int base_col = col_block * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    if (base_col >= N) return;

    const int cols_to_process = min(COLS_PER_WARP, N - base_col);
    const int num_k_blocks = K / QK4_1;

    float sums[COLS_PER_WARP];
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) sums[c] = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

        // Load and quantize activation once
        float a_block[32];
        const float* act_ptr = &activation[static_cast<int64_t>(row) * K + k_start];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
            int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        // Process all columns with this activation
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const block_q4_1* w_block = &weight[static_cast<int64_t>(col) * num_k_blocks + kb];
            const float d_w = read_half_as_float(w_block->d);
            const float m_w = read_half_as_float(w_block->m);

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_block->qs[i * 4]);
                int8_t w0 = (int8_t)(w_packed & 0x0F);
                int8_t w1 = (int8_t)((w_packed >> 8) & 0x0F);
                int8_t w2 = (int8_t)((w_packed >> 16) & 0x0F);
                int8_t w3 = (int8_t)((w_packed >> 24) & 0x0F);
                int w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                            ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);

                w0 = (int8_t)((w_packed >> 4) & 0x0F);
                w1 = (int8_t)((w_packed >> 12) & 0x0F);
                w2 = (int8_t)((w_packed >> 20) & 0x0F);
                w3 = (int8_t)((w_packed >> 28) & 0x0F);
                w_pack = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) |
                        ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }
            sums[c] += d_w * d_a * (float)sumi + m_w * a_sum;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        for (int c = 0; c < cols_to_process; ++c) {
            output[row * N + base_col + c] = sums[c];
        }
    }
}

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 16) {
        // M <= 16: Warp kernel with SM saturation
        const int total_outputs = M * N;
        const int threads_per_block = 256;

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

        const int min_blocks = num_sms * 2;
        const int needed_blocks = (total_outputs + threads_per_block / WARP_SIZE - 1) / (threads_per_block / WARP_SIZE);
        const int num_blocks = (needed_blocks > min_blocks) ? needed_blocks : min_blocks;

        gemm_warp_kernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // M > 16: Column-reuse kernel
        constexpr int COLS_PER_BLOCK = 256;
        dim3 blocks(M, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);

        gemm_column_reuse_kernel<<<blocks, 512>>>(
            reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Qwen3-4B FFN Down Q4_1 GEMM Final");
}
