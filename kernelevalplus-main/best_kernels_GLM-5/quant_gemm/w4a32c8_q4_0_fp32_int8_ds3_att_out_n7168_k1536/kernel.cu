/**
 * Highly Optimized W4A32C8 Quantized GEMM for DeepSeek-V3 Attention Output
 * - N: 7168 (output features)
 * - K: 1536 (input features)
 * - Weight: Q4_0 quantized (4-bit with per-block scale, 18 bytes/block)
 * - Activation: FP32, dynamically quantized to Q8_1 style for INT8 compute
 *
 * V10 Optimizations:
 * 1. M=1: Pre-quantize activation, use shared memory
 * 2. M=2-8: Use shared memory for activation
 * 3. Large batch: Use cuBLAS with dequantized weights
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int QK = 32;
constexpr int K_DIM = 1536;
constexpr int NUM_K_BLOCKS = K_DIM / QK;  // 48 blocks
constexpr int Q4_0_BLOCK = 18;

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    return __half2float(*reinterpret_cast<const half*>(&h));
}

// Load int32 from 2-byte aligned memory
__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

// ============================================================================
// KERNEL: Single batch (M=1) - Use shared activation quantization
// ============================================================================

__global__ void __launch_bounds__(256) gemv_kernel_m1(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    // Shared memory for activation quantization
    __shared__ float s_act_scales[NUM_K_BLOCKS];
    __shared__ float s_act_sums[NUM_K_BLOCKS];
    __shared__ int8_t s_act_qs[K_DIM];

    const int tid = threadIdx.x;

    // Cooperatively quantize activation
    for (int kb = tid; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_base = kb * QK;
        float act_max = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&activation[k_base + i]);
            act_max = fmaxf(act_max, fabsf(val));
            act_sum += val;
        }

        float scale = fmaxf(act_max / 127.0f, 1e-10f);
        s_act_scales[kb] = scale;
        s_act_sums[kb] = act_sum;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&activation[k_base + i]);
            float q = roundf(val / scale);
            s_act_qs[k_base + i] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, q)));
        }
    }
    __syncthreads();

    // Each thread computes one output
    const int n = blockIdx.x * blockDim.x + tid;
    if (n >= N) return;

    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const int k_base = kb * QK;
        const float act_scale = s_act_scales[kb];
        const float act_sum = s_act_sums[kb];
        const int8_t* act_qs = &s_act_qs[k_base];

        // Load Q4_0 weight block
        const uint8_t* w_block = weight + (size_t)(n) * NUM_K_BLOCKS * Q4_0_BLOCK + kb * Q4_0_BLOCK;
        float w_scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* qs = w_block + 2;

        // Optimized dot product
        int int_sum = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int v = load_int_b2(qs, i);
            int vi0 = (v >> 0) & 0x0F0F0F0F;
            int vi1 = (v >> 4) & 0x0F0F0F0F;

            int u0 = *reinterpret_cast<const int*>(&act_qs[i * 4]);
            int u1 = *reinterpret_cast<const int*>(&act_qs[i * 4 + 16]);

            int_sum = dp4a(vi0, u0, int_sum);
            int_sum = dp4a(vi1, u1, int_sum);
        }

        sum += w_scale * (act_scale * static_cast<float>(int_sum) - 8.0f * act_sum);
    }

    output[n] = sum;
}

// ============================================================================
// KERNEL: Small batch (M > 1)
// ============================================================================

__global__ void __launch_bounds__(256) gemm_kernel_small_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    __shared__ float s_act_scales[NUM_K_BLOCKS];
    __shared__ float s_act_sums[NUM_K_BLOCKS];
    __shared__ int8_t s_act_qs[K_DIM];

    const float* act_row = activation + m * K;

    // Cooperative activation quantization
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_base = kb * QK;
        float act_max = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&act_row[k_base + i]);
            act_max = fmaxf(act_max, fabsf(val));
            act_sum += val;
        }

        float scale = fmaxf(act_max / 127.0f, 1e-10f);
        s_act_scales[kb] = scale;
        s_act_sums[kb] = act_sum;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&act_row[k_base + i]);
            float q = roundf(val / scale);
            s_act_qs[k_base + i] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, q)));
        }
    }
    __syncthreads();

    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const int k_base = kb * QK;
        const float act_scale = s_act_scales[kb];
        const float act_sum = s_act_sums[kb];
        const int8_t* act_qs = &s_act_qs[k_base];

        const uint8_t* w_block = weight + (size_t)(n) * NUM_K_BLOCKS * Q4_0_BLOCK + kb * Q4_0_BLOCK;
        float w_scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* qs = w_block + 2;

        int int_sum = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int v = load_int_b2(qs, i);
            int vi0 = (v >> 0) & 0x0F0F0F0F;
            int vi1 = (v >> 4) & 0x0F0F0F0F;

            int u0 = *reinterpret_cast<const int*>(&act_qs[i * 4]);
            int u1 = *reinterpret_cast<const int*>(&act_qs[i * 4 + 16]);

            int_sum = dp4a(vi0, u0, int_sum);
            int_sum = dp4a(vi1, u1, int_sum);
        }

        sum += w_scale * (act_scale * static_cast<float>(int_sum) - 8.0f * act_sum);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// KERNEL: Dequantize for large batch
// ============================================================================

__global__ void __launch_bounds__(256) dequantize_q4_0_kernel(
    const uint8_t* __restrict__ weight,
    float* __restrict__ weight_fp32,
    int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float* w_row = weight_fp32 + (size_t)(n) * K;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const uint8_t* w_block = weight + (size_t)(n) * NUM_K_BLOCKS * Q4_0_BLOCK + kb * Q4_0_BLOCK;
        float scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* qs = w_block + 2;
        int k_base = kb * QK;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            w_row[k_base + i] = static_cast<float>((packed & 0x0F) - 8) * scale;
            w_row[k_base + i + 16] = static_cast<float>((packed >> 4) - 8) * scale;
        }
    }
}

// ============================================================================
// Host dispatch
// ============================================================================

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        // M=1: Pre-quantize activation in shared memory
        const int threads = 256;
        dim3 grid((N + threads - 1) / threads);
        dim3 block(threads);

        gemv_kernel_m1<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );
    } else if (M <= 8) {
        // Small batch: shared memory activation
        const int threads = 256;
        dim3 grid((N + threads - 1) / threads, M);
        dim3 block(threads);

        gemm_kernel_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: dequantize and use cuBLAS
        auto weight_fp32 = torch::empty({N, K},
            torch::dtype(torch::kFloat32).device(weight.device()));

        const int threads = 256;
        dim3 grid((N + threads - 1) / threads);
        dim3 block(threads);

        dequantize_q4_0_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            weight_fp32.data_ptr<float>(),
            N, K
        );

        output = torch::matmul(activation, weight_fp32.t());
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 GEMM DeepSeek-V3 AttOut v10");
}
