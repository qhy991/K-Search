/**
 * Optimized Quantized GEMM for DeepSeek-V3 MoE Routing Down Projection - Final
 *
 * Target: RTX 4090 (128 SMs, Ada Lovelace, CC 8.9)
 * Configuration: N=7168, K=512, M=variable
 * Format: W4A32C8 - Q4_0 weights, FP32 activations with Q8_1 style dynamic quantization
 *
 * Q4_0 block format (18 bytes):
 * - 2 bytes: scale (fp16) - "d"
 * - 16 bytes: 32 x 4-bit values packed as byte[i] = q[i] | (q[i+16] << 4)
 *
 * Q4_0 uses offset-8 encoding: values 0-15 represent -8 to +7
 * Dequantization: w = (q - 8) * scale
 *
 * GEMM formula with Q4_0 and Q8_1 activation:
 * result = d4_0 * (d8_1 * sumi - 8 * s8_1)
 *
 * Performance (tuned configurations):
 * - M=1: ~1070 GFLOPS (~99% baseline)
 * - M=2: ~1220 GFLOPS
 * - M=3: ~1265 GFLOPS
 * - M=4: ~1485 GFLOPS
 * - M=5: ~1350 GFLOPS
 * - M=8: ~1430 GFLOPS
 * - M=512: ~1740 GFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int Q4_0_BLOCK = 18;

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void load_float4(const float* ptr, float& v0, float& v1, float& v2, float& v3) {
    const float4* ptr4 = reinterpret_cast<const float4*>(ptr);
    float4 v = *ptr4;
    v0 = v.x; v1 = v.y; v2 = v.z; v3 = v.w;
}

__device__ void unpack_q4_0_weights(const uint8_t* qs_ptr, int8_t* w_vals) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t packed = qs_ptr[i];
        w_vals[i] = static_cast<int8_t>(packed & 0x0F);
        w_vals[i + 16] = static_cast<int8_t>((packed >> 4) & 0x0F);
    }
}

// ============================================================================
// M=1 Shared Memory Kernel - Optimized for memory-bound regime
// ============================================================================

__global__ void __launch_bounds__(256) gemm_q4_0_m1_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    __shared__ float s_act[512];

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Vectorized load of activation row
    const float4* act4 = reinterpret_cast<const float4*>(activation);
    float4* s_act4 = reinterpret_cast<float4*>(s_act);
    const int num_vec4 = K / 4;

    for (int idx = threadIdx.x; idx < num_vec4; idx += blockDim.x) {
        s_act4[idx] = act4[idx];
    }
    __syncthreads();

    const int outputs_per_warp = 4;
    const int total_warps = gridDim.x * num_warps;
    const int num_k_blocks = K / QK;

    for (int base_n = blockIdx.x * num_warps * outputs_per_warp + warp_id * outputs_per_warp;
         base_n < N;
         base_n += total_warps * outputs_per_warp) {

        float partial_sums[4] = {0.0f};
        bool valid[4];

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            int n = base_n + o;
            valid[o] = (n < N);
        }

        for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
            const int k_start = kb * QK;

            float a_vals[QK];
            #pragma unroll
            for (int i = 0; i < QK; i += 4) {
                load_float4(&s_act[k_start + i],
                           a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
            }

            float a_max = 0.0f;
            float a_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK; i++) {
                a_max = fmaxf(a_max, fabsf(a_vals[i]));
                a_sum += a_vals[i];
            }
            const float d_a = a_max / 127.0f;
            const float scale = 127.0f / fmaxf(a_max, 1e-10f);

            #pragma unroll
            for (int o = 0; o < 4; o++) {
                if (!valid[o]) continue;

                int n = base_n + o;

                const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK;
                const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));

                int8_t w_vals[QK];
                unpack_q4_0_weights(w_block + 2, w_vals);

                int32_t sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int a_pack = (int((uint8_t)__float2int_rn(a_vals[i*4] * scale))) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+1] * scale)) << 8) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+2] * scale)) << 16) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+3] * scale)) << 24);

                    int w_pack = (int((uint8_t)w_vals[i*4])) |
                                (int((uint8_t)w_vals[i*4+1]) << 8) |
                                (int((uint8_t)w_vals[i*4+2]) << 16) |
                                (int((uint8_t)w_vals[i*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                partial_sums[o] += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
            }
        }

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            if (!valid[o]) continue;
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                int n = base_n + o;
                if (n < N) {
                    output[n] = partial_sums[o];
                }
            }
        }
    }
}

// ============================================================================
// General Kernel for M >= 2
// ============================================================================

template<int OUTPUTS_PER_WARP>
__global__ void __launch_bounds__(256) gemm_q4_0_general_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int global_warp_id = blockIdx.x * num_warps + warp_id;
    const int total_warps = gridDim.x * num_warps;
    const int num_outputs = M * N;
    const int num_k_blocks = K / QK;

    for (int base_idx = global_warp_id * OUTPUTS_PER_WARP;
         base_idx < num_outputs;
         base_idx += total_warps * OUTPUTS_PER_WARP) {

        float partial_sums[OUTPUTS_PER_WARP] = {0.0f};
        int m_vals[OUTPUTS_PER_WARP], n_vals[OUTPUTS_PER_WARP];
        bool valid[OUTPUTS_PER_WARP];

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            int idx = base_idx + o;
            valid[o] = (idx < num_outputs);
            if (valid[o]) {
                m_vals[o] = idx / N;
                n_vals[o] = idx % N;
            }
        }

        for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
            const int k_start = kb * QK;

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (!valid[o]) continue;

                int m = m_vals[o];
                int n = n_vals[o];

                float a_vals[QK];
                #pragma unroll
                for (int i = 0; i < QK; i += 4) {
                    load_float4(&activation[m * K + k_start + i],
                               a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
                }

                float a_max = 0.0f;
                float a_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < QK; i++) {
                    a_max = fmaxf(a_max, fabsf(a_vals[i]));
                    a_sum += a_vals[i];
                }
                const float d_a = a_max / 127.0f;
                const float scale = 127.0f / fmaxf(a_max, 1e-10f);

                const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK;
                const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));

                int8_t w_vals[QK];
                unpack_q4_0_weights(w_block + 2, w_vals);

                int32_t sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int a_pack = (int((uint8_t)__float2int_rn(a_vals[i*4] * scale))) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+1] * scale)) << 8) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+2] * scale)) << 16) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+3] * scale)) << 24);

                    int w_pack = (int((uint8_t)w_vals[i*4])) |
                                (int((uint8_t)w_vals[i*4+1]) << 8) |
                                (int((uint8_t)w_vals[i*4+2]) << 16) |
                                (int((uint8_t)w_vals[i*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                partial_sums[o] += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
            }
        }

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            if (!valid[o]) continue;
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                output[m_vals[o] * N + n_vals[o]] = partial_sums[o];
            }
        }
    }
}

// ============================================================================
// Unified Forward with Strategy Dispatch - Final Version
// ============================================================================

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const int threads = 256;

    if (M == 1) {
        // Memory-bound regime: match 128 SMs for optimal scheduling
        const int blocks = 128;
        gemm_q4_0_m1_shared_kernel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    else if (M == 2) {
        const int blocks = 256;
        gemm_q4_0_general_kernel<2><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    else if (M == 3) {
        const int blocks = 384;
        gemm_q4_0_general_kernel<3><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    else if (M == 4) {
        // Best config: 512 blocks
        const int blocks = 512;
        gemm_q4_0_general_kernel<4><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    else if (M == 5) {
        const int blocks = 512;
        gemm_q4_0_general_kernel<4><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    else if (M <= 8) {
        const int blocks = 512;
        gemm_q4_0_general_kernel<4><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    else if (M <= 16) {
        const int blocks = 512;
        gemm_q4_0_general_kernel<6><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    else {
        // Compute-bound regime: maximize throughput
        const int blocks = 512;
        gemm_q4_0_general_kernel<8><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Q4_0 GEMM for DeepSeek-V3 MoE Routing Down N7168 K512 Final");
}
