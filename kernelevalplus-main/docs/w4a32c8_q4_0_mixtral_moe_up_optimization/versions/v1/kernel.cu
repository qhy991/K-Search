/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel
 *
 * Mixtral-8x7B MoE Up projection: N=14336, K=4096
 *
 * Memory Analysis:
 * - Weight: N x K/32 x 18 = 14336 x 128 x 18 = 33.0 MB
 * - Activation per row: K x 4 = 16 KB
 * - Output per row: N x 4 = 56 KB
 *
 * For M=1: Read 33 MB weights, write 56 KB -> memory bound
 * For M=512: Read 33 MB weights + 8 MB activations, write 28 MB -> memory bound
 *
 * Q4_0 Encoding:
 * - q = round(val / scale + 8), q in [0, 15]
 * - Decode: val = scale x (q - 8), where (q - 8) is in [-8, 7]
 *
 * Formula: output = activation @ (d_w x (q_w - 8))^T
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// Q4_0 block structure: 18 bytes
typedef struct {
    uint16_t d;        // FP16 scale
    uint8_t qs[16];    // Packed 4-bit values (32 values total)
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// FP16 to FP32 conversion helper (CRITICAL: use union method)
__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// ============================================================================
// Optimized kernel for N=14336, K=4096
// Uses vectorized memory access and loop unrolling
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_fp32_optimized(
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
    const float* act_row = activation + m * K;

    float sum = 0.0f;

    // Process all K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0* w_block = &w_row[kb];
        const float scale = half_to_float(w_block->d);
        const float* act_ptr = act_row + kb * 32;

        // Fully unrolled computation for better ILP
        // Q4_0 decode: val = scale * (q - 8)
        // Each qs[i] contains two 4-bit values: low nibble (pos 0-15), high nibble (pos 16-31)
        float b0 = (float)((int8_t)(w_block->qs[0] & 0x0F) - 8) * act_ptr[0]
                 + (float)((int8_t)(w_block->qs[0] >> 4) - 8) * act_ptr[16];
        float b1 = (float)((int8_t)(w_block->qs[1] & 0x0F) - 8) * act_ptr[1]
                 + (float)((int8_t)(w_block->qs[1] >> 4) - 8) * act_ptr[17];
        float b2 = (float)((int8_t)(w_block->qs[2] & 0x0F) - 8) * act_ptr[2]
                 + (float)((int8_t)(w_block->qs[2] >> 4) - 8) * act_ptr[18];
        float b3 = (float)((int8_t)(w_block->qs[3] & 0x0F) - 8) * act_ptr[3]
                 + (float)((int8_t)(w_block->qs[3] >> 4) - 8) * act_ptr[19];
        float b4 = (float)((int8_t)(w_block->qs[4] & 0x0F) - 8) * act_ptr[4]
                 + (float)((int8_t)(w_block->qs[4] >> 4) - 8) * act_ptr[20];
        float b5 = (float)((int8_t)(w_block->qs[5] & 0x0F) - 8) * act_ptr[5]
                 + (float)((int8_t)(w_block->qs[5] >> 4) - 8) * act_ptr[21];
        float b6 = (float)((int8_t)(w_block->qs[6] & 0x0F) - 8) * act_ptr[6]
                 + (float)((int8_t)(w_block->qs[6] >> 4) - 8) * act_ptr[22];
        float b7 = (float)((int8_t)(w_block->qs[7] & 0x0F) - 8) * act_ptr[7]
                 + (float)((int8_t)(w_block->qs[7] >> 4) - 8) * act_ptr[23];
        float b8 = (float)((int8_t)(w_block->qs[8] & 0x0F) - 8) * act_ptr[8]
                 + (float)((int8_t)(w_block->qs[8] >> 4) - 8) * act_ptr[24];
        float b9 = (float)((int8_t)(w_block->qs[9] & 0x0F) - 8) * act_ptr[9]
                 + (float)((int8_t)(w_block->qs[9] >> 4) - 8) * act_ptr[25];
        float b10 = (float)((int8_t)(w_block->qs[10] & 0x0F) - 8) * act_ptr[10]
                  + (float)((int8_t)(w_block->qs[10] >> 4) - 8) * act_ptr[26];
        float b11 = (float)((int8_t)(w_block->qs[11] & 0x0F) - 8) * act_ptr[11]
                  + (float)((int8_t)(w_block->qs[11] >> 4) - 8) * act_ptr[27];
        float b12 = (float)((int8_t)(w_block->qs[12] & 0x0F) - 8) * act_ptr[12]
                  + (float)((int8_t)(w_block->qs[12] >> 4) - 8) * act_ptr[28];
        float b13 = (float)((int8_t)(w_block->qs[13] & 0x0F) - 8) * act_ptr[13]
                  + (float)((int8_t)(w_block->qs[13] >> 4) - 8) * act_ptr[29];
        float b14 = (float)((int8_t)(w_block->qs[14] & 0x0F) - 8) * act_ptr[14]
                  + (float)((int8_t)(w_block->qs[14] >> 4) - 8) * act_ptr[30];
        float b15 = (float)((int8_t)(w_block->qs[15] & 0x0F) - 8) * act_ptr[15]
                  + (float)((int8_t)(w_block->qs[15] >> 4) - 8) * act_ptr[31];

        sum += scale * (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 +
                        b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// PyTorch binding
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // blockDim.x * blockDim.y must be <= 1024
    dim3 block(256);  // 256 threads per block
    dim3 grid((N + 255) / 256, M);

    gemm_q4_0_fp32_optimized<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM forward pass");
}
