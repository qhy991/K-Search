/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel
 * v12 - Fixed vectorized loads with correct index mapping
 *
 * Q4_0 layout: Each byte qs[i] contains:
 *   - low nibble (bits 0-3): position i (0-15)
 *   - high nibble (bits 4-7): position i+16 (16-31)
 *
 * Float4 layout:
 *   v0: positions 0-3   (v0.x=0, v0.y=1, v0.z=2, v0.w=3)
 *   v1: positions 4-7
 *   v2: positions 8-11
 *   v3: positions 12-15
 *   v4: positions 16-19 (v4.x=16, v4.y=17, v4.z=18, v4.w=19)
 *   ...
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

__global__ void __launch_bounds__(64) gemm_q4_0_fp32_vec64(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    const float4* act_row_vec = (const float4*)(activation + m * K);

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0 w_block = w_row[kb];
        const float scale = half_to_float(w_block.d);

        const float4* act_vec = act_row_vec + kb * 8;

        float4 v0 = act_vec[0];  // 0,1,2,3
        float4 v1 = act_vec[1];  // 4,5,6,7
        float4 v2 = act_vec[2];  // 8,9,10,11
        float4 v3 = act_vec[3];  // 12,13,14,15
        float4 v4 = act_vec[4];  // 16,17,18,19
        float4 v5 = act_vec[5];  // 20,21,22,23
        float4 v6 = act_vec[6];  // 24,25,26,27
        float4 v7 = act_vec[7];  // 28,29,30,31

        // qs[0]: low->pos 0=v0.x, high->pos 16=v4.x
        float b0 = (float)((int8_t)(w_block.qs[0] & 0x0F) - 8) * v0.x
                 + (float)((int8_t)(w_block.qs[0] >> 4) - 8) * v4.x;
        // qs[1]: low->pos 1=v0.y, high->pos 17=v4.y
        float b1 = (float)((int8_t)(w_block.qs[1] & 0x0F) - 8) * v0.y
                 + (float)((int8_t)(w_block.qs[1] >> 4) - 8) * v4.y;
        // qs[2]: low->pos 2=v0.z, high->pos 18=v4.z
        float b2 = (float)((int8_t)(w_block.qs[2] & 0x0F) - 8) * v0.z
                 + (float)((int8_t)(w_block.qs[2] >> 4) - 8) * v4.z;
        // qs[3]: low->pos 3=v0.w, high->pos 19=v4.w
        float b3 = (float)((int8_t)(w_block.qs[3] & 0x0F) - 8) * v0.w
                 + (float)((int8_t)(w_block.qs[3] >> 4) - 8) * v4.w;
        // qs[4]: low->pos 4=v1.x, high->pos 20=v5.x
        float b4 = (float)((int8_t)(w_block.qs[4] & 0x0F) - 8) * v1.x
                 + (float)((int8_t)(w_block.qs[4] >> 4) - 8) * v5.x;
        // qs[5]: low->pos 5=v1.y, high->pos 21=v5.y
        float b5 = (float)((int8_t)(w_block.qs[5] & 0x0F) - 8) * v1.y
                 + (float)((int8_t)(w_block.qs[5] >> 4) - 8) * v5.y;
        // qs[6]: low->pos 6=v1.z, high->pos 22=v5.z
        float b6 = (float)((int8_t)(w_block.qs[6] & 0x0F) - 8) * v1.z
                 + (float)((int8_t)(w_block.qs[6] >> 4) - 8) * v5.z;
        // qs[7]: low->pos 7=v1.w, high->pos 23=v5.w
        float b7 = (float)((int8_t)(w_block.qs[7] & 0x0F) - 8) * v1.w
                 + (float)((int8_t)(w_block.qs[7] >> 4) - 8) * v5.w;
        // qs[8]: low->pos 8=v2.x, high->pos 24=v6.x
        float b8 = (float)((int8_t)(w_block.qs[8] & 0x0F) - 8) * v2.x
                 + (float)((int8_t)(w_block.qs[8] >> 4) - 8) * v6.x;
        // qs[9]: low->pos 9=v2.y, high->pos 25=v6.y
        float b9 = (float)((int8_t)(w_block.qs[9] & 0x0F) - 8) * v2.y
                 + (float)((int8_t)(w_block.qs[9] >> 4) - 8) * v6.y;
        // qs[10]: low->pos 10=v2.z, high->pos 26=v6.z
        float b10 = (float)((int8_t)(w_block.qs[10] & 0x0F) - 8) * v2.z
                  + (float)((int8_t)(w_block.qs[10] >> 4) - 8) * v6.z;
        // qs[11]: low->pos 11=v2.w, high->pos 27=v6.w
        float b11 = (float)((int8_t)(w_block.qs[11] & 0x0F) - 8) * v2.w
                  + (float)((int8_t)(w_block.qs[11] >> 4) - 8) * v6.w;
        // qs[12]: low->pos 12=v3.x, high->pos 28=v7.x
        float b12 = (float)((int8_t)(w_block.qs[12] & 0x0F) - 8) * v3.x
                  + (float)((int8_t)(w_block.qs[12] >> 4) - 8) * v7.x;
        // qs[13]: low->pos 13=v3.y, high->pos 29=v7.y
        float b13 = (float)((int8_t)(w_block.qs[13] & 0x0F) - 8) * v3.y
                  + (float)((int8_t)(w_block.qs[13] >> 4) - 8) * v7.y;
        // qs[14]: low->pos 14=v3.z, high->pos 30=v7.z
        float b14 = (float)((int8_t)(w_block.qs[14] & 0x0F) - 8) * v3.z
                  + (float)((int8_t)(w_block.qs[14] >> 4) - 8) * v7.z;
        // qs[15]: low->pos 15=v3.w, high->pos 31=v7.w
        float b15 = (float)((int8_t)(w_block.qs[15] & 0x0F) - 8) * v3.w
                  + (float)((int8_t)(w_block.qs[15] >> 4) - 8) * v7.w;

        sum += scale * (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 +
                        b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15);
    }

    output[m * N + n] = sum;
}

__global__ void __launch_bounds__(256) gemm_q4_0_fp32_vec256(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    const float4* act_row_vec = (const float4*)(activation + m * K);

    float sum = 0.0f;

    #pragma unroll 4
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0 w_block = w_row[kb];
        const float scale = half_to_float(w_block.d);

        const float4* act_vec = act_row_vec + kb * 8;

        float4 v0 = act_vec[0];
        float4 v1 = act_vec[1];
        float4 v2 = act_vec[2];
        float4 v3 = act_vec[3];
        float4 v4 = act_vec[4];
        float4 v5 = act_vec[5];
        float4 v6 = act_vec[6];
        float4 v7 = act_vec[7];

        float b0 = (float)((int8_t)(w_block.qs[0] & 0x0F) - 8) * v0.x
                 + (float)((int8_t)(w_block.qs[0] >> 4) - 8) * v4.x;
        float b1 = (float)((int8_t)(w_block.qs[1] & 0x0F) - 8) * v0.y
                 + (float)((int8_t)(w_block.qs[1] >> 4) - 8) * v4.y;
        float b2 = (float)((int8_t)(w_block.qs[2] & 0x0F) - 8) * v0.z
                 + (float)((int8_t)(w_block.qs[2] >> 4) - 8) * v4.z;
        float b3 = (float)((int8_t)(w_block.qs[3] & 0x0F) - 8) * v0.w
                 + (float)((int8_t)(w_block.qs[3] >> 4) - 8) * v4.w;
        float b4 = (float)((int8_t)(w_block.qs[4] & 0x0F) - 8) * v1.x
                 + (float)((int8_t)(w_block.qs[4] >> 4) - 8) * v5.x;
        float b5 = (float)((int8_t)(w_block.qs[5] & 0x0F) - 8) * v1.y
                 + (float)((int8_t)(w_block.qs[5] >> 4) - 8) * v5.y;
        float b6 = (float)((int8_t)(w_block.qs[6] & 0x0F) - 8) * v1.z
                 + (float)((int8_t)(w_block.qs[6] >> 4) - 8) * v5.z;
        float b7 = (float)((int8_t)(w_block.qs[7] & 0x0F) - 8) * v1.w
                 + (float)((int8_t)(w_block.qs[7] >> 4) - 8) * v5.w;
        float b8 = (float)((int8_t)(w_block.qs[8] & 0x0F) - 8) * v2.x
                 + (float)((int8_t)(w_block.qs[8] >> 4) - 8) * v6.x;
        float b9 = (float)((int8_t)(w_block.qs[9] & 0x0F) - 8) * v2.y
                 + (float)((int8_t)(w_block.qs[9] >> 4) - 8) * v6.y;
        float b10 = (float)((int8_t)(w_block.qs[10] & 0x0F) - 8) * v2.z
                  + (float)((int8_t)(w_block.qs[10] >> 4) - 8) * v6.z;
        float b11 = (float)((int8_t)(w_block.qs[11] & 0x0F) - 8) * v2.w
                  + (float)((int8_t)(w_block.qs[11] >> 4) - 8) * v6.w;
        float b12 = (float)((int8_t)(w_block.qs[12] & 0x0F) - 8) * v3.x
                  + (float)((int8_t)(w_block.qs[12] >> 4) - 8) * v7.x;
        float b13 = (float)((int8_t)(w_block.qs[13] & 0x0F) - 8) * v3.y
                  + (float)((int8_t)(w_block.qs[13] >> 4) - 8) * v7.y;
        float b14 = (float)((int8_t)(w_block.qs[14] & 0x0F) - 8) * v3.z
                  + (float)((int8_t)(w_block.qs[14] >> 4) - 8) * v7.z;
        float b15 = (float)((int8_t)(w_block.qs[15] & 0x0F) - 8) * v3.w
                  + (float)((int8_t)(w_block.qs[15] >> 4) - 8) * v7.w;

        sum += scale * (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 +
                        b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M <= 8) {
        dim3 block(64);
        dim3 grid((N + 63) / 64, M);
        gemm_q4_0_fp32_vec64<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_q4_0_fp32_vec256<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM vectorized");
}
