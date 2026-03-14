/**
 * Highly Optimized Q4_0 GEMM for DeepSeek-V2 MoE routing up
 * - N: 5120 (output features)
 * - K: 1536 (input features)
 * - Weight: Q4_0 quantized (llama.cpp format: 18 bytes/block)
 * - Activation: FP32, dynamically quantized to INT8 per block
 *
 * Target: Close to baseline 2.47 TFLOPS for M=1
 *
 * Key Optimizations:
 * - Optimized block count for RTX 4090 (128 SMs)
 * - Warp-level K-splitting with efficient parallelism
 * - LDG for read-only cache utilization
 * - Multiple outputs per warp for better efficiency
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int Q4_0_BLOCK = 18;
constexpr int NUM_K_BLOCKS = 48;  // K=1536 / 32 = 48

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

// Optimized block dot product with vectorized loads
__device__ __forceinline__ float dot_q4_0_block(
    const uint8_t* __restrict__ w_block,
    const float* __restrict__ a_ptr
) {
    float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));

    // Vectorized load of activations
    float a_vals[QK];
    float a_max = 0.0f;
    float a_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < QK; i += 4) {
        float4 v = __ldg(reinterpret_cast<const float4*>(&a_ptr[i]));
        a_vals[i] = v.x;
        a_vals[i+1] = v.y;
        a_vals[i+2] = v.z;
        a_vals[i+3] = v.w;
    }

    #pragma unroll
    for (int i = 0; i < QK; i++) {
        a_max = fmaxf(a_max, fabsf(a_vals[i]));
        a_sum += a_vals[i];
    }

    const float scale = 127.0f / fmaxf(a_max, 1e-10f);
    float d_a = a_max / 127.0f;

    const uint8_t* qs = w_block + 2;
    int32_t sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t b0 = __ldg(&qs[i * 4 + 0]);
        uint8_t b1 = __ldg(&qs[i * 4 + 1]);
        uint8_t b2 = __ldg(&qs[i * 4 + 2]);
        uint8_t b3 = __ldg(&qs[i * 4 + 3]);

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

// Optimized warp kernel with 8 outputs per warp for better utilization
__global__ void __launch_bounds__(256) gemm_q4_0_warp_kernel(
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

    // Each warp handles 8 outputs (reduced from 4 for better work distribution)
    constexpr int OUTPUTS_PER_WARP = 8;

    for (int base_idx = global_warp_id * OUTPUTS_PER_WARP; base_idx < M * N; base_idx += total_warps * OUTPUTS_PER_WARP) {
        float partial_sums[OUTPUTS_PER_WARP] = {0.0f};

        int m_vals[OUTPUTS_PER_WARP], n_vals[OUTPUTS_PER_WARP];
        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            int idx = base_idx + o;
            if (idx >= M * N) { m_vals[o] = -1; n_vals[o] = -1; continue; }
            m_vals[o] = idx / N;
            n_vals[o] = idx % N;
        }

        // Each lane processes a subset of K blocks
        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const int k_start = kb * QK;

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (m_vals[o] < 0) continue;
                int m = m_vals[o];
                int n = n_vals[o];

                const uint8_t* w_block = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
                const float* a_ptr = activation + static_cast<int64_t>(m) * K + k_start;

                partial_sums[o] += dot_q4_0_block(w_block, a_ptr);
            }
        }

        // Warp reduction
        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                int idx = base_idx + o;
                if (idx < M * N) {
                    output[idx] = partial_sums[o];
                }
            }
        }
    }
}

// Fallback kernel for large M
__global__ void __launch_bounds__(256) gemm_q4_0_fallback_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    float sum = 0.0f;
    const float* act_row = activation + static_cast<int64_t>(m) * K;

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

    if (M <= 64) {
        // Optimized block count: enough to cover all outputs with good parallelism
        // For M=1, N=5120: 5120/8 = 640 warps needed
        // With 8 warps per block, need 80 blocks minimum
        // Use 256 blocks for better SM utilization (128 SMs * 2)
        const int threads = 256;
        const int warps_per_block = threads / WARP_SIZE;
        constexpr int outputs_per_warp = 8;
        const int total_outputs = M * N;
        const int warps_needed = (total_outputs + outputs_per_warp - 1) / outputs_per_warp;
        const int blocks = min(256, (warps_needed + warps_per_block - 1) / warps_per_block);
        const int final_blocks = max(128, blocks);  // At least 128 for full SM coverage

        gemm_q4_0_warp_kernel<<<final_blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_q4_0_fallback_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM v7 - Optimized");
}
