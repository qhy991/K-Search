/**
 * Optimized Quantized GEMM for DeepSeek-V3 MoE Down Projection - v3
 * Configuration: N=7168, K=18432, M=variable
 * Format: W8A32C8 - Q8_0 weights, FP32 activations
 *
 * v3 Strategy: Unified kernel with optimal parameters from v1/v2
 * - Use v1 parameters for M=1-4 (best performance)
 * - Fine-tune M=5 with different opw
 * - Keep M=8, M=512 optimized
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

template<int OUTPUTS_PER_WARP>
__global__ void __launch_bounds__(256)
gemm_q8_0_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * num_warps + warp_id;
    const int total_warps = gridDim.x * num_warps;
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
                #pragma unroll
                for (int i = 0; i < QK; i += 4) {
                    float4 v = *reinterpret_cast<const float4*>(act_ptr + i);
                    a_vals[i] = v.x;
                    a_vals[i+1] = v.y;
                    a_vals[i+2] = v.z;
                    a_vals[i+3] = v.w;
                }

                float a_max = 0.0f;
                #pragma unroll
                for (int i = 0; i < QK; i++) {
                    a_max = fmaxf(a_max, fabsf(a_vals[i]));
                }
                const float scale = 127.0f / fmaxf(a_max, 1e-10f);
                const float d_a = a_max / 127.0f;

                int32_t sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int a_pack = 0;
                    a_pack |= (int((uint8_t)__float2int_rn(a_vals[i*4] * scale)));
                    a_pack |= (int((uint8_t)__float2int_rn(a_vals[i*4+1] * scale)) << 8);
                    a_pack |= (int((uint8_t)__float2int_rn(a_vals[i*4+2] * scale)) << 16);
                    a_pack |= (int((uint8_t)__float2int_rn(a_vals[i*4+3] * scale)) << 24);

                    int w_pack = 0;
                    w_pack |= (int((uint8_t)wb->qs[i*4]));
                    w_pack |= (int((uint8_t)wb->qs[i*4+1]) << 8);
                    w_pack |= (int((uint8_t)wb->qs[i*4+2]) << 16);
                    w_pack |= (int((uint8_t)wb->qs[i*4+3]) << 24);

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

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const int threads = 256;

    // Optimized parameters from v1/v2 experiments
    int blocks, opw;

    if (M == 1) {
        blocks = 512;
        opw = 1;
    } else if (M == 2) {
        blocks = 512;  // v1 setting
        opw = 2;
    } else if (M == 3) {
        blocks = 512;  // v1 setting
        opw = 3;
    } else if (M == 4) {
        blocks = 512;  // v1 setting
        opw = 4;
    } else if (M == 5) {
        blocks = 512;
        opw = 5;  // Try matching M
    } else if (M <= 8) {
        blocks = 512;
        opw = 4;
    } else if (M <= 16) {
        blocks = 512;
        opw = 6;
    } else {
        blocks = 512;
        opw = 8;
    }

    #define LAUNCH_KERNEL(OPW) \
        gemm_q8_0_kernel<OPW><<<blocks, threads>>>( \
            weight.data_ptr<uint8_t>(), \
            activation.data_ptr<float>(), \
            output.data_ptr<float>(), \
            M, N, K \
        )

    switch (opw) {
        case 1: LAUNCH_KERNEL(1); break;
        case 2: LAUNCH_KERNEL(2); break;
        case 3: LAUNCH_KERNEL(3); break;
        case 4: LAUNCH_KERNEL(4); break;
        case 5: LAUNCH_KERNEL(5); break;
        case 6: LAUNCH_KERNEL(6); break;
        default: LAUNCH_KERNEL(8); break;
    }

    #undef LAUNCH_KERNEL

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM for DeepSeek-V3 MoE Down v3");
}
