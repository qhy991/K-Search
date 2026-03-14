#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// BLOCK_Q8_0 structure: 2 bytes scale + 32 bytes int8 values = 34 bytes per block
struct __align__(2) block_q8_0 {
    uint16_t d;        // scale stored as FP16 bits
    int8_t qs[32];     // quantized int8 values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// FP16 to FP32 conversion helper
__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// INT8 dot product using DP4A (dot product accumulate 4 pairs)
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c));
    return result;
}

// ============================================================================
// Kernel for M=1: Direct computation without split-K for memory efficiency
// Each thread block computes a contiguous chunk of outputs
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q8_0_m1_direct(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K
) {
    const int n_base = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;

    // Each thread computes multiple output elements (strided)
    for (int n = n_base; n < N; n += stride) {
        float sum = 0.0f;

        for (int kb = 0; kb < num_blocks; kb++) {
            const int w_block_idx = n * num_blocks + kb;

            // Load weight scale
            const uint16_t w_d = __ldg(&w_blocks[w_block_idx].d);
            const float scale_w = read_half_as_float(w_d);

            // Load quantized weights (32 int8 values) - load as bytes to avoid alignment issues
            int8_t w_qs[32];
            const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(weight);
            const int w_block_offset = w_block_idx * 34 + 2;  // +2 to skip the scale
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                w_qs[i] = __ldg(reinterpret_cast<const int8_t*>(w_bytes + w_block_offset + i));
            }

            // Load activation block (32 floats)
            const float* act_ptr = activation + kb * 32;

            // Compute amax for activation quantization
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));
                amax = fmaxf(amax, fabsf(v.x));
                amax = fmaxf(amax, fabsf(v.y));
                amax = fmaxf(amax, fabsf(v.z));
                amax = fmaxf(amax, fabsf(v.w));
            }

            const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

            // Quantize activation and compute dot product
            int sumi = 0;

            // Process 8 vectors of 4 floats each
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));

                // Quantize 4 values
                int a0 = max(-128, min(127, __float2int_rn(v.x * inv_scale_a)));
                int a1 = max(-128, min(127, __float2int_rn(v.y * inv_scale_a)));
                int a2 = max(-128, min(127, __float2int_rn(v.z * inv_scale_a)));
                int a3 = max(-128, min(127, __float2int_rn(v.w * inv_scale_a)));

                // Pack into int32
                int a_packed = (a3 << 24) | ((a2 & 0xFF) << 16) | ((a1 & 0xFF) << 8) | (a0 & 0xFF);

                // Pack weights - load 4 int8 values
                const int idx = i * 4;
                int w_packed = (w_qs[idx + 3] << 24) | ((w_qs[idx + 2] & 0xFF) << 16) |
                              ((w_qs[idx + 1] & 0xFF) << 8) | (w_qs[idx] & 0xFF);

                sumi = dp4a(w_packed, a_packed, sumi);
            }

            sum += scale_w * scale_a * (float)sumi;
        }

        output[n] = sum;
    }
}

// ============================================================================
// Kernel for small batches (2 <= M <= 8): Simple row-parallel approach
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q8_0_small_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks; kb++) {
        const int w_block_idx = n * num_blocks + kb;
        const uint16_t w_d = __ldg(&w_blocks[w_block_idx].d);
        const float scale_w = read_half_as_float(w_d);

        // Load weight quantized values
        int8_t w_qs[32];
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(weight);
        const int w_block_offset = w_block_idx * 34 + 2;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            w_qs[i] = __ldg(reinterpret_cast<const int8_t*>(w_bytes + w_block_offset + i));
        }

        const float* act_ptr = activation + m * K + kb * 32;

        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
        }

        const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));

            int a0 = max(-128, min(127, __float2int_rn(v.x * inv_scale_a)));
            int a1 = max(-128, min(127, __float2int_rn(v.y * inv_scale_a)));
            int a2 = max(-128, min(127, __float2int_rn(v.z * inv_scale_a)));
            int a3 = max(-128, min(127, __float2int_rn(v.w * inv_scale_a)));

            int a_packed = (a3 << 24) | ((a2 & 0xFF) << 16) | ((a1 & 0xFF) << 8) | (a0 & 0xFF);

            const int idx = i * 4;
            int w_packed = (w_qs[idx + 3] << 24) | ((w_qs[idx + 2] & 0xFF) << 16) |
                          ((w_qs[idx + 1] & 0xFF) << 8) | (w_qs[idx] & 0xFF);

            sumi = dp4a(w_packed, a_packed, sumi);
        }

        sum += scale_w * scale_a * (float)sumi;
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Kernel for large batches (M >= 16): Tiled approach with shared memory
// ============================================================================
constexpr int TILE_M = 16;
constexpr int TILE_N = 32;
constexpr int TILE_K_BLOCKS = 4;

__global__ void __launch_bounds__(512) gemm_q8_0_large_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int m = block_m + warp;
    const int n = block_n + lane;

    const bool valid = (m < M) && (n < N);

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks = K / 32;

    __shared__ float s_weight_scales[TILE_K_BLOCKS][TILE_N];
    __shared__ int s_weight_qs[TILE_K_BLOCKS][TILE_N][8];  // Store 8 packed ints

    float sum = 0.0f;

    const int num_iterations = (num_blocks + TILE_K_BLOCKS - 1) / TILE_K_BLOCKS;

    for (int iter = 0; iter < num_iterations; iter++) {
        const int kb_start = iter * TILE_K_BLOCKS;
        const int kb_end = min(kb_start + TILE_K_BLOCKS, num_blocks);
        const int actual_k_blocks = kb_end - kb_start;

        // Load weight blocks into shared memory
        #pragma unroll
        for (int k = 0; k < TILE_K_BLOCKS; k++) {
            if (k < actual_k_blocks && warp == 0) {
                const int kb = kb_start + k;
                const int load_n = block_n + lane;
                if (load_n < N) {
                    const int w_block_idx = load_n * num_blocks + kb;
                    const uint16_t w_d = __ldg(&w_blocks[w_block_idx].d);
                    s_weight_scales[k][lane] = read_half_as_float(w_d);

                    // Load weight quantized values
                    const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(weight);
                    const int w_block_offset = w_block_idx * 34 + 2;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int idx = i * 4;
                        // Need to sign-extend int8 values
                        int8_t w0 = reinterpret_cast<const int8_t*>(w_bytes + w_block_offset)[idx];
                        int8_t w1 = reinterpret_cast<const int8_t*>(w_bytes + w_block_offset)[idx + 1];
                        int8_t w2 = reinterpret_cast<const int8_t*>(w_bytes + w_block_offset)[idx + 2];
                        int8_t w3 = reinterpret_cast<const int8_t*>(w_bytes + w_block_offset)[idx + 3];
                        int w_packed = (w3 << 24) | ((w2 & 0xFF) << 16) | ((w1 & 0xFF) << 8) | (w0 & 0xFF);
                        s_weight_qs[k][lane][i] = w_packed;
                    }
                }
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K_BLOCKS; k++) {
            if (k >= actual_k_blocks) break;

            const int kb = kb_start + k;

            if (valid) {
                const float* act_ptr = activation + m * K + kb * 32;

                float amax = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float4 v = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));
                    amax = fmaxf(amax, fabsf(v.x));
                    amax = fmaxf(amax, fabsf(v.y));
                    amax = fmaxf(amax, fabsf(v.z));
                    amax = fmaxf(amax, fabsf(v.w));
                }

                const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

                const float scale_w = s_weight_scales[k][lane];

                int sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float4 v = __ldg(reinterpret_cast<const float4*>(act_ptr + i * 4));

                    int a0 = max(-128, min(127, __float2int_rn(v.x * inv_scale_a)));
                    int a1 = max(-128, min(127, __float2int_rn(v.y * inv_scale_a)));
                    int a2 = max(-128, min(127, __float2int_rn(v.z * inv_scale_a)));
                    int a3 = max(-128, min(127, __float2int_rn(v.w * inv_scale_a)));

                    int a_packed = (a3 << 24) | ((a2 & 0xFF) << 16) | ((a1 & 0xFF) << 8) | (a0 & 0xFF);
                    int w_packed = s_weight_qs[k][lane][i];

                    sumi = dp4a(w_packed, a_packed, sumi);
                }

                sum += scale_w * scale_a * (float)sumi;
            }
        }

        __syncthreads();
    }

    if (valid) {
        output[m * N + n] = sum;
    }
}

// ============================================================================
// PyTorch binding with strategy dispatch
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M == 1) {
        // M=1: Use direct computation without split-K for memory efficiency
        // Use enough blocks to fully utilize GPU
        const int threads_per_block = 128;
        const int num_blocks = min(128, (N + threads_per_block - 1) / threads_per_block);

        dim3 block(threads_per_block);
        dim3 grid(num_blocks);

        gemm_q8_0_m1_direct<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );
    }
    else if (M <= 8) {
        // Small batch: Simple row-parallel
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);

        gemm_q8_0_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
    else {
        // Large batch: Tiled with shared memory
        dim3 block(512);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        gemm_q8_0_large_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W8A32C8 Q8_0 x FP32_INT8 GEMM forward pass");
}
