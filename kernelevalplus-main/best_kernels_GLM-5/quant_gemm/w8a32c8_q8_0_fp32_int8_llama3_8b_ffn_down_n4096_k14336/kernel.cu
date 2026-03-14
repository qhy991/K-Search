/**
 * W8A32C8 Quantized GEMM for LLaMA-3-8B FFN Down Projection - v13
 *
 * Parameters: N = 4096, K = 14336, M = batch size
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 *
 * ============================================================================
 * PERFORMANCE ANALYSIS (from previous versions)
 * ============================================================================
 *
 * Best performance so far:
 *   v8:  M=1: 1.70 TFLOPS, M=512: 1.94 TFLOPS
 *   v12: M=1: 1.74 TFLOPS, M=512: 1.81 TFLOPS
 *
 * Key observations:
 *   1. v8 uses OUTPUTS_PER_WARP=6 with 2048 blocks for M=512
 *   2. More blocks = better occupancy = better performance for large M
 *   3. For small M, fewer outputs per warp is better
 *
 * Strategy for v13:
 *   - Combine the best approaches from v8 and v12
 *   - Use OUTPUTS_PER_WARP=1 for M<=8 (better for memory-bound)
 *   - Use OUTPUTS_PER_WARP=4 with high block count for M>8
 *   - Ensure maximum occupancy for large M
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

__device__ __forceinline__ int dp4a_device(int a, int b, int c) {
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
// Unified kernel: each warp computes OUTPUTS_PER_WARP outputs
// ============================================================================
template<int OUTPUTS_PER_WARP>
__launch_bounds__(256)
__global__ void gemm_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int global_warp_id = blockIdx.x * 8 + warp_id;
    const int total_warps = gridDim.x * 8;
    const int num_blocks_k = K / QK;

    for (int base_idx = global_warp_id * OUTPUTS_PER_WARP;
         base_idx < M * N;
         base_idx += total_warps * OUTPUTS_PER_WARP) {

        float partial_sums[OUTPUTS_PER_WARP];
        int m_vals[OUTPUTS_PER_WARP];
        int n_vals[OUTPUTS_PER_WARP];
        bool valid[OUTPUTS_PER_WARP];

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            partial_sums[o] = 0.0f;
            int idx = base_idx + o;
            valid[o] = (idx < M * N);
            m_vals[o] = valid[o] ? (idx / N) : 0;
            n_vals[o] = valid[o] ? (idx % N) : 0;
        }

        for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (!valid[o]) continue;

                const int m = m_vals[o];
                const int n = n_vals[o];

                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + size_t(n) * num_blocks_k * sizeof(block_q8_0) + kb * sizeof(block_q8_0)
                );

                const float d_w = __half2float(wb->d);
                const int k_start = kb * QK;
                const float* act_ptr = activation + size_t(m) * K + k_start;

                float a_vals[QK];
                float a_max = 0.0f;

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float4 v = *reinterpret_cast<const float4*>(act_ptr + i * 4);
                    a_vals[i*4] = v.x;
                    a_vals[i*4+1] = v.y;
                    a_vals[i*4+2] = v.z;
                    a_vals[i*4+3] = v.w;
                    a_max = fmaxf(a_max, fabsf(v.x));
                    a_max = fmaxf(a_max, fabsf(v.y));
                    a_max = fmaxf(a_max, fabsf(v.z));
                    a_max = fmaxf(a_max, fabsf(v.w));
                }

                const float d_a = a_max / 127.0f;
                const float inv_d_a = (a_max > 0.0f) ? (127.0f / a_max) : 0.0f;

                int32_t sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int a_pack = (int((uint8_t)__float2int_rn(a_vals[i*4] * inv_d_a))) |
                                 (int((uint8_t)__float2int_rn(a_vals[i*4+1] * inv_d_a)) << 8) |
                                 (int((uint8_t)__float2int_rn(a_vals[i*4+2] * inv_d_a)) << 16) |
                                 (int((uint8_t)__float2int_rn(a_vals[i*4+3] * inv_d_a)) << 24);

                    int w_pack = (int((uint8_t)wb->qs[i*4])) |
                                 (int((uint8_t)wb->qs[i*4+1]) << 8) |
                                 (int((uint8_t)wb->qs[i*4+2]) << 16) |
                                 (int((uint8_t)wb->qs[i*4+3]) << 24);

                    sumi = dp4a_device(a_pack, w_pack, sumi);
                }

                partial_sums[o] += d_w * d_a * float(sumi);
            }
        }

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            if (!valid[o]) continue;
            float sum = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                output[m_vals[o] * N + n_vals[o]] = sum;
            }
        }
    }
}

// ============================================================================
// Host entry point with strategy dispatch
// ============================================================================
torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const int threads = 256;
    const int SM_COUNT = 128;

    // Calculate optimal number of blocks based on M
    // For small M: fewer blocks, outputs_per_warp=1
    // For large M: more blocks, outputs_per_warp=4

    #define LAUNCH_KERNEL(OPW, NUM_BLOCKS) \
        gemm_kernel<OPW><<<NUM_BLOCKS, threads>>>( \
            weight.data_ptr<uint8_t>(), \
            activation.data_ptr<float>(), \
            output.data_ptr<float>(), \
            M, N, K \
        )

    if (M == 1) {
        LAUNCH_KERNEL(1, 512);
    } else if (M == 2) {
        LAUNCH_KERNEL(2, 512);
    } else if (M == 3) {
        LAUNCH_KERNEL(3, 512);
    } else if (M == 4) {
        LAUNCH_KERNEL(4, 512);
    } else if (M == 5) {
        LAUNCH_KERNEL(5, 512);
    } else if (M == 6) {
        LAUNCH_KERNEL(6, 512);
    } else if (M == 7) {
        LAUNCH_KERNEL(7, 512);
    } else if (M == 8) {
        LAUNCH_KERNEL(4, 512);
    } else if (M <= 16) {
        LAUNCH_KERNEL(6, 768);
    } else if (M <= 32) {
        LAUNCH_KERNEL(6, 1024);
    } else if (M <= 64) {
        LAUNCH_KERNEL(6, 1536);
    } else if (M <= 128) {
        LAUNCH_KERNEL(6, 1792);
    } else if (M <= 256) {
        LAUNCH_KERNEL(6, 2048);
    } else {
        // M >= 512: maximum parallelism
        LAUNCH_KERNEL(6, 2048);
    }

    #undef LAUNCH_KERNEL

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM for LLaMA-3-8B FFN Down v13");
}
