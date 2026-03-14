/**
 * Q8_0 x FP32 GEMM Kernel for DeepSeek-V2 LM Head
 * N=102400, K=5120 (very large output dimension)
 *
 * V8: Using exact best kernel from reference
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// Q8_0 block structure: 34 bytes (scale + 32 int8 values)
typedef struct {
    uint16_t d;        // FP16 scale (stored as raw bits)
    int8_t qs[32];     // quantized values
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// FP16 conversion helper
__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Safe pack 4 int8 values to int32 without alignment issues
__device__ __forceinline__ int pack_int4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int)a & 0xFF) | (((int)b & 0xFF) << 8) | (((int)c & 0xFF) << 16) | (((int)d & 0xFF) << 24);
}

/**
 * M=1 kernel with 32 threads (one warp) - maximum occupancy
 */
__global__ void __launch_bounds__(32) gemm_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K
) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    if (n >= N) return;

    const int num_blocks = K / 32;
    const block_q8_0* w_blocks = (const block_q8_0*)weight;

    double sum = 0.0;

    // Grid-stride loop over K blocks
    for (int kb = tid; kb < num_blocks; kb += blockDim.x) {
        // Load weight block
        const block_q8_0 w_block = w_blocks[n * num_blocks + kb];
        const float d_w = read_half_as_float(w_block.d);

        // Load activation block with vectorized loads
        const float* act_block = activation + kb * 32;
        const float4* act_vec = reinterpret_cast<const float4*>(act_block);

        // Compute activation scale
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = act_vec[i];
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
        }
        const float d_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        // DP4A dot product
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Pack weight values safely
            int w_packed = pack_int4(
                w_block.qs[i * 4 + 0],
                w_block.qs[i * 4 + 1],
                w_block.qs[i * 4 + 2],
                w_block.qs[i * 4 + 3]
            );

            float4 v = act_vec[i];
            int8_t a_vals[4];
            a_vals[0] = (int8_t)max(-128, min(127, __float2int_rn(v.x * inv_d_a)));
            a_vals[1] = (int8_t)max(-128, min(127, __float2int_rn(v.y * inv_d_a)));
            a_vals[2] = (int8_t)max(-128, min(127, __float2int_rn(v.z * inv_d_a)));
            a_vals[3] = (int8_t)max(-128, min(127, __float2int_rn(v.w * inv_d_a)));

            int a_packed = pack_int4(a_vals[0], a_vals[1], a_vals[2], a_vals[3]);
            sumi = __dp4a(w_packed, a_packed, sumi);
        }

        sum += (double)d_w * (double)d_a * (double)sumi;
    }

    // Warp shuffle reduction (no shared memory needed)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only thread 0 writes the result
    if (tid == 0) {
        output[n] = (float)sum;
    }
}

/**
 * Small batch kernel (M=2 to M=16) with DP4A
 */
__global__ void __launch_bounds__(256) gemm_small_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const int num_blocks = K / 32;
    const block_q8_0* w_blocks = (const block_q8_0*)weight;
    const float* act_row = activation + m * K;

    double sum = 0.0;

    for (int kb = 0; kb < num_blocks; kb++) {
        const block_q8_0 w_block = w_blocks[n * num_blocks + kb];
        const float d_w = read_half_as_float(w_block.d);

        const float* act_block = act_row + kb * 32;
        const float4* act_vec = reinterpret_cast<const float4*>(act_block);

        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = act_vec[i];
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
        }
        const float d_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_packed = pack_int4(
                w_block.qs[i * 4 + 0],
                w_block.qs[i * 4 + 1],
                w_block.qs[i * 4 + 2],
                w_block.qs[i * 4 + 3]
            );

            float4 v = act_vec[i];
            int8_t a_vals[4];
            a_vals[0] = (int8_t)max(-128, min(127, __float2int_rn(v.x * inv_d_a)));
            a_vals[1] = (int8_t)max(-128, min(127, __float2int_rn(v.y * inv_d_a)));
            a_vals[2] = (int8_t)max(-128, min(127, __float2int_rn(v.z * inv_d_a)));
            a_vals[3] = (int8_t)max(-128, min(127, __float2int_rn(v.w * inv_d_a)));

            int a_packed = pack_int4(a_vals[0], a_vals[1], a_vals[2], a_vals[3]);
            sumi = __dp4a(w_packed, a_packed, sumi);
        }

        sum += (double)d_w * (double)d_a * (double)sumi;
    }

    output[m * N + n] = (float)sum;
}

/**
 * Dequantize Q8_0 weights to FP16 for large batch processing
 */
__global__ void dequantize_q8_0_to_fp16_kernel(
    const uint8_t* __restrict__ weight_q,
    half* __restrict__ weight_fp16,
    const int N, const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = blockIdx.y;

    const int num_blocks = K / 32;
    const block_q8_0* w_blocks = (const block_q8_0*)weight_q;

    if (n >= N) return;

    const block_q8_0 w_block = w_blocks[n * num_blocks + kb];

    union { uint16_t u16; half f16; } un;
    un.u16 = w_block.d;
    const half d_w = un.f16;

    const int k_start = kb * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        weight_fp16[n * K + k_start + i] = __hmul(__int2half_rn((int)w_block.qs[i]), d_w);
    }
}

/**
 * Forward pass with adaptive strategy selection
 */
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");

    const int num_blocks = K / 32;

    if (M == 1) {
        // Single token: use 32 threads (one warp) per block for maximum occupancy
        auto output = torch::empty({1, N}, torch::dtype(torch::kFloat32).device(weight.device()));

        dim3 block(32);
        dim3 grid(N);

        gemm_m1_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );

        return output;
    }
    else if (M <= 16) {
        // Small batch: per-row kernel
        auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, M);

        gemm_small_batch_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );

        return output;
    }
    else {
        // Large batch: dequantize and use FP16 Tensor Core via PyTorch
        auto weight_fp16 = torch::empty({N, K}, torch::dtype(torch::kFloat16).device(weight.device()));

        dim3 dequant_block(256);
        dim3 dequant_grid((N + 255) / 256, num_blocks);

        dequantize_q8_0_to_fp16_kernel<<<dequant_grid, dequant_block>>>(
            weight.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(weight_fp16.data_ptr()),
            N, K
        );

        auto activation_fp16 = activation.to(torch::kFloat16);
        return activation_fp16.matmul(weight_fp16.t()).to(torch::kFloat32);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 x FP32 GEMM forward pass");
}
