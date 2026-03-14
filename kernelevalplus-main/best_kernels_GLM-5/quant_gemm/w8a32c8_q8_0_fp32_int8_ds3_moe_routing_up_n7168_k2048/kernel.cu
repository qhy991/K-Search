/**
 * Optimized Quantized GEMM for DeepSeek-V3 MoE Routing Up Projection
 *
 * Target: RTX 4090 (128 SMs, Ada Lovelace, CC 8.9)
 * Configuration: N=7168, K=2048, M=variable
 * Format: W8A32C8 - Q8_0 weights, FP32 activations
 *
 * V11: 128 blocks (1024 warps), close to minimum needed (896 warps)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int NUM_K_BLOCKS = 64;

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

__global__ void __launch_bounds__(256) gemm_q8_0_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K, int row_offset
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / WARP_SIZE;

    __shared__ float act_scales[NUM_K_BLOCKS];
    __shared__ int8_t act_qs[NUM_K_BLOCKS * QK];

    for (int kb = tid; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_start = kb * QK;
        float a_vals[QK];
        #pragma unroll
        for (int i = 0; i < QK; i += 4) {
            load_float4(&activation[(size_t)row_offset * K + k_start + i], a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
        }
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) a_max = fmaxf(a_max, fabsf(a_vals[i]));
        const float scale = 127.0f / fmaxf(a_max, 1e-10f);
        act_scales[kb] = a_max / 127.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) act_qs[kb * QK + i] = (int8_t)__float2int_rn(a_vals[i] * scale);
    }
    __syncthreads();

    constexpr int OUTPUTS_PER_WARP = 8;
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    const int total_warps = gridDim.x * (blockDim.x / WARP_SIZE);

    for (int n_base = global_warp_id * OUTPUTS_PER_WARP; n_base < N; n_base += total_warps * OUTPUTS_PER_WARP) {
        float partial_sums[OUTPUTS_PER_WARP] = {0.0f};

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const float d_a = act_scales[kb];
            const int8_t* a_qs = &act_qs[kb * QK];
            int a_packed[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_packed[i] = (int((uint8_t)a_qs[i*4])) |
                             (int((uint8_t)a_qs[i*4+1]) << 8) |
                             (int((uint8_t)a_qs[i*4+2]) << 16) |
                             (int((uint8_t)a_qs[i*4+3]) << 24);
            }

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                const int n = n_base + o;
                if (n >= N) continue;
                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (size_t)(n * NUM_K_BLOCKS + kb) * sizeof(block_q8_0));
                const float d_w = read_half_as_float(wb->d);
                int32_t sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int w_pack = (int((uint8_t)wb->qs[i*4])) |
                                (int((uint8_t)wb->qs[i*4+1]) << 8) |
                                (int((uint8_t)wb->qs[i*4+2]) << 16) |
                                (int((uint8_t)wb->qs[i*4+3]) << 24);
                    sumi = dp4a(a_packed[i], w_pack, sumi);
                }
                partial_sums[o] += d_w * d_a * (float)sumi;
            }
        }

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            const int n = n_base + o;
            if (n < N) {
                partial_sums[o] = warp_reduce_sum(partial_sums[o]);
                if (lane_id == 0) output[(size_t)row_offset * N + n] = partial_sums[o];
            }
        }
    }
}

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
            if (valid[o]) { m_vals[o] = idx / N; n_vals[o] = idx % N; }
        }

        for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (!valid[o]) continue;
                int m = m_vals[o], n = n_vals[o];
                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (size_t)(n * num_blocks_k + kb) * sizeof(block_q8_0));
                const float d_w = read_half_as_float(wb->d);
                const int k_start = kb * QK;
                float a_vals[QK];
                #pragma unroll
                for (int i = 0; i < QK; i += 4) {
                    load_float4(&activation[(size_t)m * K + k_start + i], a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
                }
                float a_max = 0.0f;
                #pragma unroll
                for (int i = 0; i < QK; i++) a_max = fmaxf(a_max, fabsf(a_vals[i]));
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
            if (lane_id == 0) output[(size_t)m_vals[o] * N + n_vals[o]] = partial_sums[o];
        }
    }
}

constexpr int TILE_M_LARGE = 4;
constexpr int TILE_N_LARGE = 64;
constexpr int BLOCK_SIZE_LARGE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE_LARGE) gemm_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int tid = threadIdx.x;
    const int num_k_blocks = K / QK;
    const int m_base = blockIdx.y * TILE_M_LARGE;
    const int n_base = blockIdx.x * TILE_N_LARGE;
    const int m_local = tid / TILE_N_LARGE;
    const int n_local = tid % TILE_N_LARGE;
    const int m = m_base + m_local;
    const int n = n_base + n_local;
    const bool valid_m = m < M;
    const bool valid_n = n < N;

    __shared__ float s_act[TILE_M_LARGE][QK];
    __shared__ int8_t s_weight[TILE_N_LARGE][QK];
    __shared__ float s_w_scale[TILE_N_LARGE];
    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_start = kb * QK;
        for (int i = tid; i < TILE_M_LARGE * QK; i += BLOCK_SIZE_LARGE) {
            int mi = i / QK, ki = i % QK;
            if (m_base + mi < M) s_act[mi][ki] = activation[(size_t)(m_base + mi) * K + k_start + ki];
        }
        for (int i = tid; i < TILE_N_LARGE; i += BLOCK_SIZE_LARGE) {
            if (n_base + i < N) {
                const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                    weight + (size_t)((n_base + i) * num_k_blocks + kb) * sizeof(block_q8_0));
                s_w_scale[i] = read_half_as_float(wb->d);
                #pragma unroll
                for (int j = 0; j < QK; j++) s_weight[i][j] = wb->qs[j];
            }
        }
        __syncthreads();

        if (valid_m && valid_n) {
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK; i++) a_max = fmaxf(a_max, fabsf(s_act[m_local][i]));
            const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t qa0 = (int8_t)__float2int_rn(s_act[m_local][i*4] / d_a);
                int8_t qa1 = (int8_t)__float2int_rn(s_act[m_local][i*4+1] / d_a);
                int8_t qa2 = (int8_t)__float2int_rn(s_act[m_local][i*4+2] / d_a);
                int8_t qa3 = (int8_t)__float2int_rn(s_act[m_local][i*4+3] / d_a);
                int a_pack = (int((uint8_t)qa0)) | (int((uint8_t)qa1) << 8) | (int((uint8_t)qa2) << 16) | (int((uint8_t)qa3) << 24);
                int w_pack = (int((uint8_t)s_weight[n_local][i*4])) | (int((uint8_t)s_weight[n_local][i*4+1]) << 8) |
                             (int((uint8_t)s_weight[n_local][i*4+2]) << 16) | (int((uint8_t)s_weight[n_local][i*4+3]) << 24);
                sumi = dp4a(a_pack, w_pack, sumi);
            }
            sum += s_w_scale[n_local] * d_a * (float)sumi;
        }
        __syncthreads();
    }
    if (valid_m && valid_n) output[(size_t)m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));
    const int threads = 256;

    if (M == 1) {
        // 128 blocks * 8 warps = 1024 warps (14% over min 896)
        gemm_q8_0_m1_kernel<<<128, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), N, K, 0);
    } else if (M == 2) {
        gemm_q8_0_m1_kernel<<<128, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), N, K, 0);
        gemm_q8_0_m1_kernel<<<128, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), N, K, 1);
    } else if (M <= 32) {
        const int gen_blocks = min(max((M * N + 31) / 32, 128), 1024);
        gemm_q8_0_general_kernel<4><<<gen_blocks, threads>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        dim3 grid((N + TILE_N_LARGE - 1) / TILE_N_LARGE, (M + TILE_M_LARGE - 1) / TILE_M_LARGE);
        gemm_large_m_kernel<<<grid, BLOCK_SIZE_LARGE>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 GEMM v11");
}
