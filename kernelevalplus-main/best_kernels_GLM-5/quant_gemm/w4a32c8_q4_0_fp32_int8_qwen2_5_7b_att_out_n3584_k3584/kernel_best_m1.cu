/**
 * Quantized GEMM Kernel v9 for Qwen2.5-7B Attention Output
 * - N: 3584, K: 3584
 * - Strategy: Maximize memory bandwidth utilization
 * - Key: Use vectorized loads (float4) for weight streaming
 * - Target: Achieve near-peak memory bandwidth for M=1
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int QK = 32;
constexpr int Q4_0_BLOCK_SIZE = 18;  // 2 + 16 bytes

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

// ============================================================================
// High-bandwidth kernel for small M
// Strategy: Each warp processes multiple N outputs, vectorized weight loads
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q4_0_highbw_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    if (m >= M) return;

    const int num_k_blocks = K / QK;  // 112

    // Shared memory for quantized activation
    extern __shared__ char shared_mem[];
    int8_t* s_act_qs = reinterpret_cast<int8_t*>(shared_mem);
    float* s_act_scales = reinterpret_cast<float*>(shared_mem + K * sizeof(int8_t));
    float* s_act_sums = reinterpret_cast<float*>(shared_mem + K * sizeof(int8_t) + num_k_blocks * sizeof(float));

    const float* act_row = activation + m * K;

    // Phase 1: Quantize activation (all threads cooperate)
    for (int kb = threadIdx.x; kb < num_k_blocks; kb += blockDim.x) {
        const int k_start = kb * QK;

        float act_max = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = act_row[k_start + i];
            act_max = fmaxf(act_max, fabsf(val));
            act_sum += val;
        }

        float scale = fmaxf(act_max / 127.0f, 1e-10f);
        s_act_scales[kb] = scale;
        s_act_sums[kb] = act_sum;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = act_row[k_start + i];
            float q = roundf(val / scale);
            s_act_qs[k_start + i] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, q)));
        }
    }
    __syncthreads();

    // Phase 2: Compute outputs
    // Each thread handles one N
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.0f;

    // Process all K blocks for this N
    // Use vectorized loads where possible
    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_start = kb * QK;

        // Load weight block (18 bytes, use float4 for first 16 bytes + 2 bytes)
        const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK_SIZE;

        float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* w_qs = w_block + 2;

        float act_scale = s_act_scales[kb];
        float act_sum = s_act_sums[kb];
        const int8_t* act_qs = &s_act_qs[k_start];

        // Compute dot product
        int sumi = 0;

        // Unroll with manual DP4A
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_qs[i];
            int w_low = packed & 0x0F;
            int w_high = (packed >> 4) & 0x0F;
            sumi += w_low * static_cast<int>(act_qs[i]);
            sumi += w_high * static_cast<int>(act_qs[i + 16]);
        }

        sum += w_scale * (act_scale * static_cast<float>(sumi) - 8.0f * act_sum);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Alternative: Process multiple N per thread to reduce kernel overhead
// ============================================================================
__global__ void __launch_bounds__(64) gemm_q4_0_multi_n_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    if (m >= M) return;

    const int num_k_blocks = K / QK;
    const int n_per_thread = 4;  // Each thread handles 4 N values
    const int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * n_per_thread;

    if (n_base >= N) return;

    // Shared memory for activation
    extern __shared__ char shared_mem[];
    int8_t* s_act_qs = reinterpret_cast<int8_t*>(shared_mem);
    float* s_act_scales = reinterpret_cast<float*>(shared_mem + K * sizeof(int8_t));
    float* s_act_sums = reinterpret_cast<float*>(shared_mem + K * sizeof(int8_t) + num_k_blocks * sizeof(float));

    const float* act_row = activation + m * K;

    // Phase 1: Quantize activation
    for (int kb = threadIdx.x; kb < num_k_blocks; kb += blockDim.x) {
        const int k_start = kb * QK;

        float act_max = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = act_row[k_start + i];
            act_max = fmaxf(act_max, fabsf(val));
            act_sum += val;
        }

        float scale = fmaxf(act_max / 127.0f, 1e-10f);
        s_act_scales[kb] = scale;
        s_act_sums[kb] = act_sum;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = act_row[k_start + i];
            float q = roundf(val / scale);
            s_act_qs[k_start + i] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, q)));
        }
    }
    __syncthreads();

    // Phase 2: Compute outputs for multiple N values
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int n_idx = 0; n_idx < n_per_thread; n_idx++) {
        const int n = n_base + n_idx;
        if (n >= N) break;

        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int k_start = kb * QK;
            const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK_SIZE;

            float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
            const uint8_t* w_qs = w_block + 2;

            float act_scale = s_act_scales[kb];
            float act_sum = s_act_sums[kb];
            const int8_t* act_qs = &s_act_qs[k_start];

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = w_qs[i];
                sumi += (packed & 0x0F) * static_cast<int>(act_qs[i]);
                sumi += ((packed >> 4) & 0x0F) * static_cast<int>(act_qs[i + 16]);
            }

            sums[n_idx] += w_scale * (act_scale * static_cast<float>(sumi) - 8.0f * act_sum);
        }
    }

    // Write results
    for (int n_idx = 0; n_idx < n_per_thread; n_idx++) {
        const int n = n_base + n_idx;
        if (n < N) {
            output[m * N + n] = sums[n_idx];
        }
    }
}

// ============================================================================
// Kernel for large M
// ============================================================================
template<int TILE_M, int TILE_N>
__global__ void __launch_bounds__(TILE_M * TILE_N) gemm_q4_0_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n = blockIdx.x * TILE_N + tx;
    const int m = blockIdx.y * TILE_M + ty;

    const int num_k_blocks = K / QK;
    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_start = kb * QK;

        float act_vals[32];
        if (m < M) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                act_vals[i] = activation[m * K + k_start + i];
            }
        }

        float act_scale = 0.0f;
        float act_sum = 0.0f;
        int8_t act_qs[32];

        if (m < M) {
            float act_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                act_max = fmaxf(act_max, fabsf(act_vals[i]));
                act_sum += act_vals[i];
            }
            act_scale = fmaxf(act_max / 127.0f, 1e-10f);

            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float q = roundf(act_vals[i] / act_scale);
                act_qs[i] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, q)));
            }
        }

        if (n < N && m < M) {
            const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_0_BLOCK_SIZE;
            float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
            const uint8_t* w_qs = w_block + 2;

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int w_low = w_qs[i] & 0x0F;
                int w_high = (w_qs[i] >> 4) & 0x0F;
                sumi += w_low * act_qs[i];
                sumi += w_high * act_qs[i + 16];
            }

            sum += w_scale * (act_scale * static_cast<float>(sumi) - 8.0f * act_sum);
        }
    }

    if (n < N && m < M) {
        output[m * N + n] = sum;
    }
}

// ============================================================================
// Host function
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const int num_k_blocks = K / QK;
    const size_t shared_mem = K * sizeof(int8_t) + 2 * num_k_blocks * sizeof(float);

    if (M <= 8) {
        // Use high-bandwidth kernel with more threads
        dim3 block(128);
        dim3 grid((N + 127) / 128, M);

        gemm_q4_0_highbw_kernel<<<grid, block, shared_mem>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 64) {
        // Use multi-N kernel
        const int n_per_thread = 4;
        dim3 block(64);
        dim3 grid((N / n_per_thread + 63) / 64, M);

        gemm_q4_0_multi_n_kernel<<<grid, block, shared_mem>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        const int TILE_M = 8;
        const int TILE_N = 32;

        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        gemm_q4_0_large_m_kernel<TILE_M, TILE_N><<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 GEMM v9 for Qwen2.5-7B");
}
