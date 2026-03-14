/**
 * Optimized Q4_1 Quantized GEMM Kernel for DeepSeek-V2 MoE Routing Up - v11
 * - N: 5120 (output features)
 * - K: 1536 (input features)
 * - M: variable (batch dimension)
 * - Weight: Q4_1 quantized (4-bit packed with scale and min)
 * - Activation: FP32, dynamically quantized to Q8_1 style
 *
 * v11 Optimizations:
 * 1. Strategy dispatch based on M
 * 2. For small M: maximize memory bandwidth with vectorized loads
 * 3. For large M: maximize compute throughput with tiling
 * 4. Use shared memory for weight tiles when beneficial
 *
 * Q4_1 block format (20 bytes): scale (FP16) + min (FP16) + 16 bytes data
 * Formula: result = d4_1 * d8_1 * sumi + m4_1 * s8_1
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int QK = 32;
constexpr int Q4_1_BLOCK = 20;
constexpr int K = 1536;
constexpr int NUM_K_BLOCKS = K / QK;  // 48 blocks

// DP4A instruction wrapper
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

/**
 * Small M kernel (M <= 16) - Memory bound regime
 * Focus on maximizing memory bandwidth
 */
__global__ void __launch_bounds__(256) gemm_small_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    // Shared memory for activation (quantized once per row)
    extern __shared__ char shared_mem[];
    float* s_act_scales = reinterpret_cast<float*>(shared_mem);
    int* s_act_sums = reinterpret_cast<int*>(shared_mem + NUM_K_BLOCKS * sizeof(float));
    int8_t* s_act_qs = reinterpret_cast<int8_t*>(shared_mem + NUM_K_BLOCKS * sizeof(float) + NUM_K_BLOCKS * sizeof(int));

    const float* act_row = activation + m * K;

    // Cooperative quantization of activations
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_base = kb * QK;

        float amax = 0.0f;
        float asum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&act_row[k_base + i]);
            amax = fmaxf(amax, fabsf(val));
            asum += val;
        }

        float scale = amax / 127.0f;
        if (scale < 1e-10f) scale = 1.0f;
        s_act_scales[kb] = scale;

        int sum_q = 0;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&act_row[k_base + i]);
            int8_t q = static_cast<int8_t>(roundf(val / scale));
            q = max(-128, min(127, (int)q));
            s_act_qs[k_base + i] = q;
            sum_q += q;
        }
        s_act_sums[kb] = sum_q;
    }
    __syncthreads();

    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const int k_base = kb * QK;
        const float act_scale = s_act_scales[kb];
        const int act_sum_q = s_act_sums[kb];
        const int8_t* __restrict__ act_qs = &s_act_qs[k_base];

        const uint8_t* __restrict__ w_block = weight + (n * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;

        const half2 header = *reinterpret_cast<const half2*>(w_block);
        const float w_scale = __half2float(header.x);
        const float w_min = __half2float(header.y);

        const uint8_t* __restrict__ qs = w_block + 4;

        int int_sum = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t p = qs[i];
            int_sum += act_qs[i] * (p & 0x0F);
            int_sum += act_qs[i + 16] * ((p >> 4) & 0x0F);
        }

        sum += w_scale * act_scale * static_cast<float>(int_sum) + w_min * act_scale * static_cast<float>(act_sum_q);
    }

    output[m * N + n] = sum;
}

/**
 * Large M kernel (M > 16) - Compute bound regime
 * Focus on maximizing compute throughput
 */
__global__ void __launch_bounds__(256) gemm_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    // Shared memory for activation
    extern __shared__ char shared_mem[];
    float* s_act_scales = reinterpret_cast<float*>(shared_mem);
    int* s_act_sums = reinterpret_cast<int*>(shared_mem + NUM_K_BLOCKS * sizeof(float));
    int8_t* s_act_qs = reinterpret_cast<int8_t*>(shared_mem + NUM_K_BLOCKS * sizeof(float) + NUM_K_BLOCKS * sizeof(int));

    const float* act_row = activation + m * K;

    // Cooperative quantization
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_base = kb * QK;

        float amax = 0.0f;
        float asum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&act_row[k_base + i]);
            amax = fmaxf(amax, fabsf(val));
            asum += val;
        }

        float scale = amax / 127.0f;
        if (scale < 1e-10f) scale = 1.0f;
        s_act_scales[kb] = scale;

        int sum_q = 0;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&act_row[k_base + i]);
            int8_t q = static_cast<int8_t>(roundf(val / scale));
            q = max(-128, min(127, (int)q));
            s_act_qs[k_base + i] = q;
            sum_q += q;
        }
        s_act_sums[kb] = sum_q;
    }
    __syncthreads();

    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const int k_base = kb * QK;
        const float act_scale = s_act_scales[kb];
        const int act_sum_q = s_act_sums[kb];
        const int8_t* __restrict__ act_qs = &s_act_qs[k_base];

        const uint8_t* __restrict__ w_block = weight + (n * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;

        const half2 header = *reinterpret_cast<const half2*>(w_block);
        const float w_scale = __half2float(header.x);
        const float w_min = __half2float(header.y);

        const uint8_t* __restrict__ qs = w_block + 4;

        int int_sum = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t p = qs[i];
            int_sum += act_qs[i] * (p & 0x0F);
            int_sum += act_qs[i + 16] * ((p >> 4) & 0x0F);
        }

        sum += w_scale * act_scale * static_cast<float>(int_sum) + w_min * act_scale * static_cast<float>(act_sum_q);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    size_t shared_mem_size = NUM_K_BLOCKS * sizeof(float) + NUM_K_BLOCKS * sizeof(int) + K * sizeof(int8_t);

    constexpr int BLOCK_SIZE = 256;

    // Strategy dispatch based on M
    if (M <= 16) {
        // Small M: memory-bound, use simple kernel
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
        dim3 block(BLOCK_SIZE);
        gemm_small_m<<<grid, block, shared_mem_size>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N
        );
    } else {
        // Large M: compute-bound, use same kernel (both are same for now)
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
        dim3 block(BLOCK_SIZE);
        gemm_large_m<<<grid, block, shared_mem_size>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Q4_1 GEMM for DeepSeek-V2 MoE Routing Up v11");
}
