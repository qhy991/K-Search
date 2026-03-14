/**
 * Quantized GEMM for Qwen2.5-7B FFN Up Projection with Q4_1 Weights - V1
 * Shape: N=18944, K=3584, Block Size=32
 *
 * Roofline Analysis:
 * - M=1:  OI ~ 2.67 FLOPs/Byte (MEMORY-BOUND, below ridge ~82)
 * - M=512: OI ~ 223 FLOPs/Byte (COMPUTE-BOUND, above ridge)
 *
 * Strategy: Single kernel optimized for both regimes
 * - Use vectorized loads for weight fetching
 * - DP4A for efficient INT8 dot products
 * - Warp-level reduction for K dimension
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

// Q4_1 block format: 16 bytes quantized data + 2 bytes scale + 2 bytes min = 20 bytes
typedef struct {
    uint16_t d;      // scale (FP16)
    uint16_t m;      // min (FP16)
    uint8_t qs[16];  // 32 x 4-bit values packed into 16 bytes
} block_q4_1;
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size must be 20 bytes");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

/**
 * Kernel for small batch sizes (M <= 8)
 * Focus on memory efficiency - each warp handles multiple output columns
 */
__global__ void __launch_bounds__(256)
gemm_q4_1_small_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    const int col_chunk = blockIdx.y;

    if (row >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    constexpr int WARPS_PER_BLOCK = 256 / WARP_SIZE;  // 8
    constexpr int COLS_PER_WARP = 8;  // Each warp handles 8 output columns
    constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * COLS_PER_WARP;  // 64

    const int base_col = col_chunk * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    if (base_col >= N) return;

    const int cols_to_process = min(COLS_PER_WARP, N - base_col);
    const int num_k_blocks = K / QK;  // 3584 / 32 = 112

    float sums[COLS_PER_WARP];
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) sums[c] = 0.0f;

    // Each lane processes a subset of K blocks
    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK;

        // Load activation block (32 floats)
        float a_block[QK];
        const float* act_ptr = &activation[row * K + k_start];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        // Compute activation quantization parameters (Q8_1 style)
        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float s_a = a_sum;  // sum for Q4_1 formula

        // Quantize activation to INT8
        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4] / d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int((uint8_t)q0)) | (int((uint8_t)q1) << 8) |
                          (int((uint8_t)q2) << 16) | (int((uint8_t)q3) << 24);
        }

        // Process each output column
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const block_q4_1* wb = reinterpret_cast<const block_q4_1*>(
                weight + (col * num_k_blocks + kb) * sizeof(block_q4_1)
            );

            const float d_w = half_to_float(wb->d);
            const float m_w = half_to_float(wb->m);
            const uint8_t* qs = wb->qs;

            int32_t sumi = 0;

            // Process lower 4 bits (first 16 values)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int w_pack = (int(qs[i * 4] & 0x0F)) |
                            (int(qs[i * 4 + 1] & 0x0F) << 8) |
                            (int(qs[i * 4 + 2] & 0x0F) << 16) |
                            (int(qs[i * 4 + 3] & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            // Process upper 4 bits (last 16 values)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int w_pack = (int((qs[i * 4] >> 4) & 0x0F)) |
                            (int((qs[i * 4 + 1] >> 4) & 0x0F) << 8) |
                            (int((qs[i * 4 + 2] >> 4) & 0x0F) << 16) |
                            (int((qs[i * 4 + 3] >> 4) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            // Q4_1 dequant formula: result = d_w * d_a * sumi + m_w * s_a
            sums[c] += d_w * d_a * (float)sumi + m_w * s_a;
        }
    }

    // Warp reduction across K blocks
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        for (int c = 0; c < cols_to_process; ++c) {
            output[row * N + base_col + c] = sums[c];
        }
    }
}

/**
 * Kernel for large batch sizes (M >= 16)
 * Focus on compute throughput - better parallelism across rows
 */
__global__ void __launch_bounds__(256)
gemm_q4_1_large_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.y;
    const int col_chunk = blockIdx.x;

    if (row >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    constexpr int WARPS_PER_BLOCK = 256 / WARP_SIZE;  // 8
    constexpr int COLS_PER_WARP = 8;
    constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * COLS_PER_WARP;  // 64

    const int base_col = col_chunk * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    if (base_col >= N) return;

    const int cols_to_process = min(COLS_PER_WARP, N - base_col);
    const int num_k_blocks = K / QK;

    float sums[COLS_PER_WARP];
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) sums[c] = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK;

        float a_block[QK];
        const float* act_ptr = &activation[row * K + k_start];

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
        for (int i = 0; i < QK; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float s_a = a_sum;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4] / d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int((uint8_t)q0)) | (int((uint8_t)q1) << 8) |
                          (int((uint8_t)q2) << 16) | (int((uint8_t)q3) << 24);
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const block_q4_1* wb = reinterpret_cast<const block_q4_1*>(
                weight + (col * num_k_blocks + kb) * sizeof(block_q4_1)
            );

            const float d_w = half_to_float(wb->d);
            const float m_w = half_to_float(wb->m);
            const uint8_t* qs = wb->qs;

            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int w_pack = (int(qs[i * 4] & 0x0F)) |
                            (int(qs[i * 4 + 1] & 0x0F) << 8) |
                            (int(qs[i * 4 + 2] & 0x0F) << 16) |
                            (int(qs[i * 4 + 3] & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int w_pack = (int((qs[i * 4] >> 4) & 0x0F)) |
                            (int((qs[i * 4 + 1] >> 4) & 0x0F) << 8) |
                            (int((qs[i * 4 + 2] >> 4) & 0x0F) << 16) |
                            (int((qs[i * 4 + 3] >> 4) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi + m_w * s_a;
        }
    }

    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        for (int c = 0; c < cols_to_process; ++c) {
            output[row * N + base_col + c] = sums[c];
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K)
{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int COLS_PER_BLOCK = 64;

    // Strategy dispatch based on batch size
    if (M <= 8) {
        // Small batch: row-major grid for better memory access
        dim3 blocks(M, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
        gemm_q4_1_small_batch<<<blocks, THREADS_PER_BLOCK>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: column-major grid for better parallelism
        dim3 blocks((N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK, M);
        gemm_q4_1_large_batch<<<blocks, THREADS_PER_BLOCK>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM for Qwen2.5-7B FFN Up Projection V1");
}
