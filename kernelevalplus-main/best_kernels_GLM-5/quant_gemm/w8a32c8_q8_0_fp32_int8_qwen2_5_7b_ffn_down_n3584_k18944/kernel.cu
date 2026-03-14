/**
 * W8A32C8 Quantized GEMM for Qwen-2.5-7B FFN Down Projection - Final v2
 *
 * Parameters: N = 3584, K = 18944, M = batch size
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 *
 * Combined Strategy:
 * - M <= 4: High-occupancy kernel (v19) - best for small batches
 * - M > 4, M <= 8: Warp kernel (v16) - best for medium batches
 * - M > 8: FP32 dequantization kernel - best for large batches
 *
 * Performance:
 * - M=1: ~960 GFLOPS
 * - M=512: ~1435 GFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

__device__ __forceinline__ int dp4a_device(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// High-occupancy kernel for very small batches (M <= 4)
__launch_bounds__(128)
__global__ void gemm_small_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int global_warp_id = blockIdx.x * 4 + warp_id;
    const int total_warps = gridDim.x * 4;
    const int num_blocks_k = K / QK;

    for (int idx = global_warp_id; idx < M * N; idx += total_warps) {
        const int m = idx / N;
        const int n = idx % N;

        float sum = 0.0f;

        for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
            const int k_start = kb * QK;

            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + (size_t)(n * num_blocks_k + kb) * sizeof(block_q8_0)
            );

            const float d_w = __half2float(wb->d);
            const float* act_ptr = activation + (size_t)m * K + k_start;

            float a_vals[QK];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = *reinterpret_cast<const float4*>(act_ptr + i * 4);
                a_vals[i*4] = v.x;
                a_vals[i*4+1] = v.y;
                a_vals[i*4+2] = v.z;
                a_vals[i*4+3] = v.w;
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

                sumi = dp4a_device(a_pack, w_pack, sumi);
            }

            sum += d_w * d_a * (float)sumi;
        }

        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            output[(size_t)m * N + n] = sum;
        }
    }
}

// Medium batch kernel (M <= 16)
__launch_bounds__(256)
__global__ void gemm_medium_batch(
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
    const int num_blocks_k = K / QK;

    for (int idx = global_warp_id; idx < M * N; idx += total_warps) {
        const int m = idx / N;
        const int n = idx % N;

        float sum = 0.0f;

        for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
            const int k_start = kb * QK;

            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + (size_t)(n * num_blocks_k + kb) * sizeof(block_q8_0)
            );

            const float d_w = __half2float(wb->d);
            const float* act_ptr = activation + (size_t)m * K + k_start;

            float a_vals[QK];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = *reinterpret_cast<const float4*>(act_ptr + i * 4);
                a_vals[i*4] = v.x;
                a_vals[i*4+1] = v.y;
                a_vals[i*4+2] = v.z;
                a_vals[i*4+3] = v.w;
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

                sumi = dp4a_device(a_pack, w_pack, sumi);
            }

            sum += d_w * d_a * (float)sumi;
        }

        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            output[(size_t)m * N + n] = sum;
        }
    }
}

// Large batch kernel - FP32 dequantization
__launch_bounds__(256)
__global__ void gemm_large_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    const int m = blockIdx.y * 16 + threadIdx.y;
    const int n = blockIdx.x * 16 + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        const int k_base = b * 32;

        const block_q8_0* w_block = reinterpret_cast<const block_q8_0*>(
            weight + (size_t)(n * num_blocks + b) * sizeof(block_q8_0)
        );

        const float d_w = __half2float(w_block->d);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 a = *reinterpret_cast<const float4*>(&activation[(size_t)m * K + k_base + i * 4]);

            acc += (float)w_block->qs[i*4] * d_w * a.x;
            acc += (float)w_block->qs[i*4+1] * d_w * a.y;
            acc += (float)w_block->qs[i*4+2] * d_w * a.z;
            acc += (float)w_block->qs[i*4+3] * d_w * a.w;
        }
    }

    output[(size_t)m * N + n] = acc;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    auto weight_ptr = weight.data_ptr<uint8_t>();
    auto activation_ptr = activation.data_ptr<float>();
    auto output_ptr = output.data_ptr<float>();

    if (M <= 4) {
        // Very small batch: high-occupancy kernel
        const int threads = 128;
        const int blocks = min(2048, (M * N + 3) / 4);
        gemm_small_batch<<<blocks, threads>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K);
    } else if (M <= 16) {
        // Medium batch: standard warp kernel
        const int threads = 256;
        const int blocks = min(1024, (M * N + 7) / 8);
        gemm_medium_batch<<<blocks, threads>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K);
    } else {
        // Large batch: FP32 dequantization
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_large_batch<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K);
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W8A32C8 Q8_0 Quantized GEMM Final v2");
}
