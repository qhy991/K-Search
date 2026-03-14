/**
 * Optimized Quantized GEMM for DeepSeek-V2 LM Head with Q8_0 Weights
 *
 * Parameters:
 *   - N = 102400, K = 5120, M = batch size
 *
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 * Formula: result = d_w * d_a * sum(w_i * a_i) where w_i and a_i are INT8
 *
 * Optimizations:
 *   - Better warp-level parallelism for large N
 *   - Improved memory access patterns
 *   - Based on working v1 kernel
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

// Q8_0 block structure: 34 bytes
typedef struct {
    uint16_t d;
    int8_t qs[32];
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size must be 34 bytes");

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// DP4A intrinsic for INT8 dot product
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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Optimized kernel using warp-level parallelism
 * Each warp computes one output element, distributing K computation across lanes
 */
__global__ void __launch_bounds__(256) gemm_q8_0_optimized_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    const int num_blocks = K / QK;

    // Process outputs with stride = num_warps
    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int m = idx / N;
        const int n = idx % N;

        float sum = 0.0f;

        // Each lane in the warp processes a subset of K blocks
        const int blocks_per_lane = (num_blocks + WARP_SIZE - 1) / WARP_SIZE;

        for (int b_offset = 0; b_offset < blocks_per_lane; b_offset++) {
            const int b = b_offset * WARP_SIZE + lane_id;
            if (b >= num_blocks) continue;

            // Load weight block for this (n, b) pair
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + (n * num_blocks + b) * sizeof(block_q8_0)
            );

            const float d_w = read_half_as_float(wb->d);
            const int k_start = b * QK;

            // Load activation block and compute max
            float a_block[QK];
            float a_max = 0.0f;

            #pragma unroll
            for (int i = 0; i < QK; i++) {
                a_block[i] = activation[m * K + k_start + i];
                a_max = fmaxf(a_max, fabsf(a_block[i]));
            }

            const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

            // Pack activation into INT8 - quantize using: round(a * 127 / a_max)
            int32_t a_packed[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4] * 127.0f / fmaxf(a_max, 1e-10f));
                int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] * 127.0f / fmaxf(a_max, 1e-10f));
                int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] * 127.0f / fmaxf(a_max, 1e-10f));
                int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] * 127.0f / fmaxf(a_max, 1e-10f));
                a_packed[i] = (int((uint8_t)q0)) | (int((uint8_t)q1) << 8) |
                              (int((uint8_t)q2) << 16) | (int((uint8_t)q3) << 24);
            }

            // INT8 dot product using DP4A
            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                // Pack weight int8 values
                int w_pack = (int((uint8_t)wb->qs[i * 4])) |
                             (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                             (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                             (int((uint8_t)wb->qs[i * 4 + 3]) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            // Q8_0 formula: result = d_w * d_a * sumi
            sum += d_w * d_a * (float)sumi;
        }

        // Warp reduction
        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            output[m * N + n] = sum;
        }
    }
}

/**
 * Simple kernel for larger batches - 2D grid
 */
__global__ void gemm_q8_0_simple_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;

    float sum = 0.0f;
    const int num_blocks = K / 32;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
            weight + (n * num_blocks + block_idx) * sizeof(block_q8_0)
        );
        const float d_w = read_half_as_float(wb->d);
        const int k_start = block_idx * 32;

        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float a = activation[m * K + k_start + i];
            a_max = fmaxf(a_max, fabsf(a));
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float a0 = activation[m * K + k_start + i * 4 + 0];
            float a1 = activation[m * K + k_start + i * 4 + 1];
            float a2 = activation[m * K + k_start + i * 4 + 2];
            float a3 = activation[m * K + k_start + i * 4 + 3];

            int8_t qa0 = (int8_t)__float2int_rn(a0 * 127.0f / fmaxf(a_max, 1e-10f));
            int8_t qa1 = (int8_t)__float2int_rn(a1 * 127.0f / fmaxf(a_max, 1e-10f));
            int8_t qa2 = (int8_t)__float2int_rn(a2 * 127.0f / fmaxf(a_max, 1e-10f));
            int8_t qa3 = (int8_t)__float2int_rn(a3 * 127.0f / fmaxf(a_max, 1e-10f));

            int a_pack = (int((uint8_t)qa0)) | (int((uint8_t)qa1) << 8) |
                         (int((uint8_t)qa2) << 16) | (int((uint8_t)qa3) << 24);

            int w_pack = (int((uint8_t)wb->qs[i * 4])) |
                         (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                         (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                         (int((uint8_t)wb->qs[i * 4 + 3]) << 24);

            int32_t partial_sum = 0;
            #if __CUDA_ARCH__ >= 610
            asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(partial_sum) : "r"(a_pack), "r"(w_pack), "r"(partial_sum));
            #else
            const int8_t* ap = reinterpret_cast<const int8_t*>(&a_pack);
            const int8_t* wp = reinterpret_cast<const int8_t*>(&w_pack);
            partial_sum = ap[0] * wp[0] + ap[1] * wp[1] + ap[2] * wp[2] + ap[3] * wp[3];
            #endif
            sum += d_w * d_a * (float)partial_sum;
        }
    }
    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 8) {
        // Small batch: use optimized DP4A kernel
        int threads = 256;
        int warps = threads / WARP_SIZE;
        int blocks = min(128, (M * N + warps - 1) / warps);  // Use more blocks for better parallelism
        gemm_q8_0_optimized_kernel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        // Larger batch: use simple 2D kernel
        dim3 block(64, 4);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_q8_0_simple_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Q8_0 GEMM for DeepSeek-V2 LM Head");
}
