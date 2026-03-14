/**
 * W8A32C8 Quantized GEMM for DeepSeek-V3 MoE Routing Down Projection v4
 *
 * Parameters: N = 7168, K = 512, M = batch size
 * Q8_0 Format (34 bytes): d (FP16) + qs[32] (int8)
 *
 * Based on v1 with optimizations:
 *   - Higher grid size for better memory bandwidth utilization
 *   - More blocks to saturate memory bandwidth on RTX 4090 (128 SMs)
 *   - Each warp computes one output element
 *   - DP4A for INT8 dot products
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int BLOCK_DIM = 256;

typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size must be 34 bytes");

__device__ __forceinline__ float read_half(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h; return __half2float(un.f16);
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int r; asm volatile("dp4a.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
#else
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
}
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__launch_bounds__(BLOCK_DIM)
__global__ void gemm_q8_0_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int global_warp_id = blockIdx.x * (BLOCK_DIM >> 5) + warp_id;

    const int num_blocks_k = K / QK;
    const int total_outputs = M * N;
    const int total_num_warps = gridDim.x * (BLOCK_DIM >> 5);

    // Each warp computes one output element
    for (int idx = global_warp_id; idx < total_outputs; idx += total_num_warps) {
        const int m = idx / N;
        const int n = idx % N;
        float sum = 0.0f;

        const float* act_row = activation + m * K;

        // Process K dimension in blocks of 32
        for (int block_k = lane_id; block_k < num_blocks_k; block_k += 32) {
            const int k_start = block_k * QK;

            // Load weight block (Q8_0 format)
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + (n * num_blocks_k + block_k) * sizeof(block_q8_0)
            );
            const float d_w = read_half(wb->d);

            // Load activation block using vectorized loads
            const float4* a_vec = reinterpret_cast<const float4*>(act_row + k_start);

            float a_block[32];
            float a_max = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = a_vec[i];
                a_block[i*4+0] = v.x;
                a_block[i*4+1] = v.y;
                a_block[i*4+2] = v.z;
                a_block[i*4+3] = v.w;
                a_max = fmaxf(a_max, fabsf(v.x));
                a_max = fmaxf(a_max, fabsf(v.y));
                a_max = fmaxf(a_max, fabsf(v.z));
                a_max = fmaxf(a_max, fabsf(v.w));
            }

            // Compute activation scale (Q8_1 style)
            const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

            // Quantize activation and compute dot product using DP4A
            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);

                int a_pack = (int((uint8_t)q0)) |
                             (int((uint8_t)q1) << 8) |
                             (int((uint8_t)q2) << 16) |
                             (int((uint8_t)q3) << 24);

                int w_pack = (int((uint8_t)wb->qs[i * 4 + 0])) |
                             (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                             (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                             (int((uint8_t)wb->qs[i * 4 + 3]) << 24);

                sumi = dp4a(a_pack, w_pack, sumi);
            }

            sum += d_w * d_a * (float)sumi;
        }

        // Warp-level reduction
        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            output[m * N + n] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kUInt8);
    TORCH_CHECK(activation.is_cuda() && activation.dtype() == torch::kFloat32);

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const int total_outputs = M * N;
    const int warps_per_block = BLOCK_DIM >> 5;
    const int min_blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

    // RTX 4090 has 128 SMs - use more blocks to saturate memory bandwidth
    // For memory-bound small batches, we need many blocks to hide memory latency
    int num_blocks;
    if (M == 1) {
        // M=1: 7168 outputs, need many blocks for memory-level parallelism
        num_blocks = min(512, min_blocks);
    } else if (M <= 4) {
        num_blocks = min(512, min_blocks);
    } else if (M <= 32) {
        num_blocks = min(1024, min_blocks);
    } else {
        num_blocks = min(2048, min_blocks);
    }

    gemm_q8_0_kernel<<<num_blocks, BLOCK_DIM>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W8A32C8 Q8_0 Quantized GEMM - DeepSeek-V3 MoE Routing Down v4");
}
