/**
 * Quantized GEMM for DeepSeek-V2 MoE Routing Down Projection v2
 *
 * Parameters:
 *   - N = 1536 (output dimension)
 *   - K = 5120 (hidden dimension)
 *   - M = batch size (variable)
 *
 * v2 Optimizations:
 *   - Increased outputs per warp from 4 to 8 for better utilization
 *   - Better weight access pattern: coalesced reads across warps
 *   - Reduced kernel launch overhead with larger work per warp
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int Q4_0_BLOCK = 18;
constexpr int NUM_K_BLOCKS = 160;  // K=5120 / 32

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

/**
 * Compute dot product for one Q4_0 block with FP32 activation
 * Returns: d_w * (d_a * sumi - 8 * sum_a)
 */
__device__ __forceinline__ float dot_q4_0_block(
    const uint8_t* w_block,
    const float* a_ptr
) {
    float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));

    // Load activation values
    float a_vals[QK];
    float a_max = 0.0f;
    float a_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < QK; i += 4) {
        load_float4(&a_ptr[i], a_vals[i], a_vals[i+1], a_vals[i+2], a_vals[i+3]);
    }

    #pragma unroll
    for (int i = 0; i < QK; i++) {
        a_max = fmaxf(a_max, fabsf(a_vals[i]));
        a_sum += a_vals[i];
    }

    float d_a = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;
    const float scale = 127.0f / fmaxf(a_max, 1e-10f);

    // Unpack Q4_0 and compute INT8 dot product
    const uint8_t* qs = w_block + 2;
    int32_t sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t b0 = qs[i * 4 + 0];
        uint8_t b1 = qs[i * 4 + 1];
        uint8_t b2 = qs[i * 4 + 2];
        uint8_t b3 = qs[i * 4 + 3];

        int w_pack_lo = (int(b0 & 0x0F)) |
                       (int(b1 & 0x0F) << 8) |
                       (int(b2 & 0x0F) << 16) |
                       (int(b3 & 0x0F) << 24);

        int w_pack_hi = (int((b0 >> 4) & 0x0F)) |
                       (int((b1 >> 4) & 0x0F) << 8) |
                       (int((b2 >> 4) & 0x0F) << 16) |
                       (int((b3 >> 4) & 0x0F) << 24);

        int a_pack_lo = (int((uint8_t)__float2int_rn(a_vals[i*4] * scale))) |
                       (int((uint8_t)__float2int_rn(a_vals[i*4+1] * scale)) << 8) |
                       (int((uint8_t)__float2int_rn(a_vals[i*4+2] * scale)) << 16) |
                       (int((uint8_t)__float2int_rn(a_vals[i*4+3] * scale)) << 24);

        int a_pack_hi = (int((uint8_t)__float2int_rn(a_vals[16+i*4] * scale))) |
                       (int((uint8_t)__float2int_rn(a_vals[16+i*4+1] * scale)) << 8) |
                       (int((uint8_t)__float2int_rn(a_vals[16+i*4+2] * scale)) << 16) |
                       (int((uint8_t)__float2int_rn(a_vals[16+i*4+3] * scale)) << 24);

        sumi = dp4a(a_pack_lo, w_pack_lo, sumi);
        sumi = dp4a(a_pack_hi, w_pack_hi, sumi);
    }

    return d_w * (d_a * (float)sumi - 8.0f * a_sum);
}

/**
 * v2: Process outputs in column-major order for better weight coalescing
 * Each warp processes consecutive N outputs (better weight locality)
 */
__global__ void __launch_bounds__(256) gemm_q4_0_kernel_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Global warp index
    const int global_warp_idx = blockIdx.x * num_warps + warp_id;

    // Each warp processes OUTPUTS_PER_WARP consecutive outputs
    constexpr int OUTPUTS_PER_WARP = 8;

    // Total number of warps needed
    const int total_outputs = M * N;
    const int total_warps_needed = (total_outputs + OUTPUTS_PER_WARP - 1) / OUTPUTS_PER_WARP;

    for (int warp_base = global_warp_idx; warp_base < total_warps_needed; warp_base += gridDim.x * num_warps) {
        int base_idx = warp_base * OUTPUTS_PER_WARP;

        float partial_sums[OUTPUTS_PER_WARP] = {0.0f};
        int n_vals[OUTPUTS_PER_WARP];
        int m_val = -1;

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            int idx = base_idx + o;
            if (idx >= total_outputs) {
                n_vals[o] = -1;
                continue;
            }
            int m = idx / N;
            int n = idx % N;
            if (m_val < 0) m_val = m;
            n_vals[o] = n;
        }

        if (m_val < 0) continue;

        // Each lane processes different K blocks
        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const int k_start = kb * QK;
            const float* a_ptr = activation + m_val * K + k_start;

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (n_vals[o] < 0) continue;
                int n = n_vals[o];

                const uint8_t* w_block = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
                partial_sums[o] += dot_q4_0_block(w_block, a_ptr);
            }
        }

        // Warp reduction
        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0 && n_vals[o] >= 0) {
                output[m_val * N + n_vals[o]] = partial_sums[o];
            }
        }
    }
}

/**
 * Optimized kernel for batch processing (M >= 8)
 * Each thread processes one output element
 */
__global__ void __launch_bounds__(256) gemm_q4_0_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    float sum = 0.0f;
    const float* act_row = activation + m * K;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const int k_start = kb * QK;
        const uint8_t* w_block = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;

        sum += dot_q4_0_block(w_block, act_row + k_start);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 32) {
        // Small batch: warp-level kernel
        const int threads = 256;
        const int num_sm = 128;  // RTX 4090 has 128 SMs
        const int blocks = num_sm * 2;  // 2 warps per SM for better occupancy
        gemm_q4_0_kernel_v2<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        // Large batch: one thread per output element
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_q4_0_batch_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM for DeepSeek-V2 MoE Routing Down v2");
}
