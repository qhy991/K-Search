/**
 * Quantized GEMM for Qwen2.5-7B FFN Up Projection (v1)
 *
 * Parameters: N = 18944, K = 3584, M = batch size
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 *
 * v1: Adapted from Qwen3-4B FFN Up kernel (v5)
 * Key features:
 * - Shared memory activation caching for M=1
 * - Warp-level reduction for K dimension
 * - DP4A instructions for INT8 dot product
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int NUM_K_BLOCKS = 112;  // 3584 / 32
constexpr int K_DIM = 3584;

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

// Shared memory structure for quantized activation blocks
struct quant_block {
    float scale;
    int8_t qs[QK];
};

/**
 * M=1 specialized kernel with shared memory activation caching
 */
__global__ void __launch_bounds__(256) gemm_m1_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / WARP_SIZE;

    // Shared memory for quantized activation (112 blocks for K=3584)
    __shared__ quant_block act_blocks[NUM_K_BLOCKS];
    __shared__ float act_scales[NUM_K_BLOCKS];

    // Phase 1: Cooperatively quantize activation into shared memory
    for (int kb = tid; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_start = kb * QK;

        // Load 32 floats using float4
        float a_vals[QK];
        #pragma unroll
        for (int i = 0; i < QK; i += 4) {
            load_float4(&activation[k_start + i], a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
        }

        // Find max
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
        }

        const float scale = 127.0f / fmaxf(a_max, 1e-10f);
        act_scales[kb] = a_max / 127.0f;  // d_a

        // Quantize
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            act_blocks[kb].qs[i] = (int8_t)__float2int_rn(a_vals[i] * scale);
        }
    }
    __syncthreads();

    // Phase 2: Each warp computes multiple outputs
    const int outputs_per_warp = 4;
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    const int total_warps = gridDim.x * (blockDim.x / WARP_SIZE);

    for (int n_base = global_warp_id * outputs_per_warp; n_base < N; n_base += total_warps * outputs_per_warp) {
        float partial_sums[outputs_per_warp] = {0.0f};
        int n_vals[outputs_per_warp];
        bool valid[outputs_per_warp];

        #pragma unroll
        for (int o = 0; o < outputs_per_warp; o++) {
            n_vals[o] = n_base + o;
            valid[o] = (n_vals[o] < N);
        }

        // Process K blocks with warp-level reduction
        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const float d_a = act_scales[kb];
            const int8_t* a_qs = act_blocks[kb].qs;

            #pragma unroll
            for (int o = 0; o < outputs_per_warp; o++) {
                if (!valid[o]) continue;

                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (n_vals[o] * NUM_K_BLOCKS + kb) * sizeof(block_q8_0)
                );

                const float d_w = read_half_as_float(wb->d);

                // INT8 dot product
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
        for (int o = 0; o < outputs_per_warp; o++) {
            if (!valid[o]) continue;
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                output[n_vals[o]] = partial_sums[o];
            }
        }
    }
}

/**
 * General kernel for M > 1 with warp-level K reduction
 */
template<int OUTPUTS_PER_WARP>
__global__ void __launch_bounds__(256) gemm_general_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int global_warp_id = blockIdx.x * num_warps + warp_id;
    const int total_warps = gridDim.x * num_warps;

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

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (!valid[o]) continue;

                int m = m_vals[o];
                int n = n_vals[o];

                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (n * NUM_K_BLOCKS + kb) * sizeof(block_q8_0)
                );

                const float d_w = read_half_as_float(wb->d);
                const int k_start = kb * QK;

                float a_vals[QK];
                #pragma unroll
                for (int i = 0; i < QK; i += 4) {
                    load_float4(&activation[m * K_DIM + k_start + i],
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

    if (M == 1) {
        // M=1: Use shared memory kernel
        const int blocks = 768;  // More blocks for larger N
        gemm_m1_shared_kernel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), N);
    } else if (M <= 4) {
        const int blocks = 768;
        gemm_general_kernel<4><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N);
    } else if (M <= 16) {
        const int blocks = 768;
        gemm_general_kernel<8><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N);
    } else {
        const int blocks = 768;
        gemm_general_kernel<16><<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM for Qwen2.5-7B FFN Up Projection v1");
}
