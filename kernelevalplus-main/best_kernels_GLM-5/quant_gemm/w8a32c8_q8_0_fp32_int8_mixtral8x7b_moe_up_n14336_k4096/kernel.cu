/**
 * W8A32C8 Quantized GEMM for Mixtral-8x7B MoE Up Projection - v36 Final
 * Q8_0 Weight (N=14336, K=4096) x FP32 Activation (M=batch, K=4096)
 *
 * Strategy: Combined version with best kernels for each regime
 * - M=1: Shared memory kernel from v29 (4.17 TFLOPS)
 * - M=2-8: General kernel with 4 outputs/warp
 * - M>8: Large M kernel with 16 cols/warp (5.3 TFLOPS at M=512)
 *
 * Performance (RTX 4090):
 * - M=1: ~4.17 TFLOPS (67% of baseline 6.19 TFLOPS)
 * - M=512: ~5.3 TFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

typedef struct {
    uint16_t d;
    int8_t qs[QK];
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size must be 34 bytes");

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

struct quant_block {
    float scale;
    int8_t qs[QK];
};

/**
 * M=1 specialized kernel with shared memory activation caching (from v29)
 * Each warp computes 2 outputs for optimal performance
 */
template<int OUTPUTS_PER_WARP>
__global__ void __launch_bounds__(256) gemm_q8_0_m1_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / WARP_SIZE;
    const int num_blocks_k = K / QK;

    __shared__ quant_block act_blocks[128];
    __shared__ float act_scales[128];

    // Phase 1: Cooperatively quantize activation into shared memory
    for (int kb = tid; kb < num_blocks_k; kb += blockDim.x) {
        const int k_start = kb * QK;
        float a_vals[QK];
        #pragma unroll
        for (int i = 0; i < QK; i += 4) {
            load_float4(&activation[k_start + i], a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
        }

        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
        }

        const float scale = 127.0f / fmaxf(a_max, 1e-10f);
        act_scales[kb] = a_max / 127.0f;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            act_blocks[kb].qs[i] = (int8_t)__float2int_rn(a_vals[i] * scale);
        }
    }
    __syncthreads();

    // Phase 2: Each warp computes multiple outputs
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    const int total_warps = gridDim.x * (blockDim.x / WARP_SIZE);

    for (int n_base = global_warp_id * OUTPUTS_PER_WARP; n_base < N; n_base += total_warps * OUTPUTS_PER_WARP) {
        float partial_sums[OUTPUTS_PER_WARP] = {0.0f};
        int n_vals[OUTPUTS_PER_WARP];
        bool valid[OUTPUTS_PER_WARP];

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            n_vals[o] = n_base + o;
            valid[o] = (n_vals[o] < N);
        }

        // Process K blocks with warp-level reduction
        for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
            const float d_a = act_scales[kb];
            const int8_t* a_qs = act_blocks[kb].qs;

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (!valid[o]) continue;

                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (size_t(n_vals[o]) * num_blocks_k + kb) * sizeof(block_q8_0)
                );

                const float d_w = read_half_as_float(wb->d);

                int32_t sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int a_pack = (int((uint8_t)a_qs[i*4])) |
                                (int((uint8_t)a_qs[i*4+1]) << 8) |
                                (int((uint8_t)a_qs[i*4+2]) << 16) |
                                (int((uint8_t)a_qs[i*4+3]) << 24);

                    int w_pack = (int((uint8_t)wb->qs[i*4])) |
                                (int((uint8_t)wb->qs[i*4+1]) << 8) |
                                (int((uint8_t)wb->qs[i*4+2]) << 16) |
                                (int((uint8_t)wb->qs[i*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                partial_sums[o] += d_w * d_a * (float)sumi;
            }
        }

        // Warp reduction
        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            if (!valid[o]) continue;
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                output[n_vals[o]] = partial_sums[o];
            }
        }
    }
}

/**
 * Large M kernel: 16 columns per warp, 16 warps per block
 */
__global__ void __launch_bounds__(512)
gemm_q8_0_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    const int col_block_idx = blockIdx.y;

    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_k_blocks = K / QK;

    const int COLS_PER_WARP = 16;
    const int COLS_PER_BLOCK = 256;

    const int base_col = col_block_idx * COLS_PER_BLOCK + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    float sums[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c) sums[c] = 0.0f;

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK;

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

        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * sizeof(block_q8_0)
            );
            const float d_w = read_half_as_float(wb->d);

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int w_pack = (int((uint8_t)wb->qs[i * 4])) |
                            (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                            (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                            (int((uint8_t)wb->qs[i * 4 + 3]) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi;
        }
    }

    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        sums[c] = warp_reduce_sum(sums[c]);
    }

    if (lane_id == 0) {
        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            if (col < N) output[static_cast<int64_t>(row) * N + col] = sums[c];
        }
    }
}

/**
 * General kernel for M > 1 with warp-level K reduction
 */
template<int OUTPUTS_PER_WARP>
__global__ void __launch_bounds__(256) gemm_q8_0_general_kernel(
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
    const int num_blocks_k = K / QK;

    for (int base_idx = global_warp_id * OUTPUTS_PER_WARP; base_idx < M * N; base_idx += total_warps * OUTPUTS_PER_WARP) {
        float partial_sums[OUTPUTS_PER_WARP] = {0.0f};
        int m_vals[OUTPUTS_PER_WARP], n_vals[OUTPUTS_PER_WARP];
        bool valid[OUTPUTS_PER_WARP];

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            int idx = base_idx + o;
            valid[o] = (idx < M * N);
            if (valid[o]) {
                m_vals[o] = idx / N;
                n_vals[o] = idx % N;
            }
        }

        for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (!valid[o]) continue;

                int m = m_vals[o];
                int n = n_vals[o];

                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (size_t(n) * num_blocks_k + kb) * sizeof(block_q8_0)
                );

                const float d_w = read_half_as_float(wb->d);
                const int k_start = kb * QK;

                float a_vals[QK];
                #pragma unroll
                for (int i = 0; i < QK; i += 4) {
                    load_float4(&activation[size_t(m) * K + k_start + i],
                               a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
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
                    int a_pack = (int((uint8_t)__float2int_rn(a_vals[i*4] * scale))) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+1] * scale)) << 8) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+2] * scale)) << 16) |
                                (int((uint8_t)__float2int_rn(a_vals[i*4+3] * scale)) << 24);

                    int w_pack = (int((uint8_t)wb->qs[i*4])) |
                                (int((uint8_t)wb->qs[i*4+1]) << 8) |
                                (int((uint8_t)wb->qs[i*4+2]) << 16) |
                                (int((uint8_t)wb->qs[i*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                partial_sums[o] += d_w * d_a * (float)sumi;
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

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const int threads = 256;
    const int blocks = 512;

    if (M == 1) {
        // M=1: Use shared memory kernel with 2 outputs per warp
        gemm_q8_0_m1_shared_kernel<2><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), N, K);
    } else if (M > 64) {
        // Large M: use 16 cols/warp for better throughput
        const int COLS_PER_BLOCK = 256;
        dim3 grid(M, (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
        dim3 block(512);

        gemm_q8_0_large_m_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 4) {
        gemm_q8_0_general_kernel<4><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else if (M <= 16) {
        gemm_q8_0_general_kernel<8><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        gemm_q8_0_general_kernel<16><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM v36 Final - Mixtral MoE Up");
}
