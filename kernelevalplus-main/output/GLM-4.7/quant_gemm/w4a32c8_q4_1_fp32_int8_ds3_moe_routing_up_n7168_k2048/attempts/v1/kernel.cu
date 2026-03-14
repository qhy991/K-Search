/**
 * Q4_1 × FP32 (dynamically quantized to Q8_1) GEMM Kernel
 *
 * Task: w4a32c8_q4_1_fp32_int8_ds3_moe_routing_up_n7168_k2048
 * - N = 7168 (output rows / hidden size)
 * - K = 2048 (inner dimension / routing expert dimension)
 * - M = variable (batch size: 1, 2, 3, 4, 5, 8, 512)
 *
 * Q4_1 Format (Weights):
 *   - d (2 bytes): half scale factor
 *   - m (2 bytes): half minimum offset
 *   - qs[16] (16 bytes): packed 4-bit values (2 per byte)
 *   - Total: 20 bytes for 32 values
 *
 * Dynamic Q8_1 Format (Activations):
 *   - For each block of 32 values:
 *     - d (half): scale = max(|x|) / 127.0
 *     - s (half): sum of original FP32 values
 *     - qs[32] (int8): quantized values
 *   - Total: 36 bytes for 32 values
 *
 * Dequantization Formula (Q4_1 × Q8_1):
 *   result = d4 * d8 * sumi + m4 * s8 / 4.0f
 *
 * Where:
 *   - d4: Q4_1 scale factor
 *   - m4: Q4_1 minimum offset
 *   - d8: Q8_1 scale factor
 *   - s8: Q8_1 sum of original values
 *   - sumi: dot product of quantized values (using DP4A)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Constants and Type Definitions
// ============================================================================

#define QK4_1 32
#define QK8_1 32

// Q4_1 block structure (weights)
typedef struct {
    half d;              // Scale factor (2 bytes)
    half m;              // Minimum offset (2 bytes)
    uint8_t qs[QK4_1/2]; // Packed 4-bit values (16 bytes)
} block_q4_1;

static_assert(sizeof(block_q4_1) == 20, "block_q4_1 must be 20 bytes");

// Q8_1 block structure (activations - computed dynamically)
typedef struct {
    half2 ds;         // d (scale) and s (sum) packed as half2 (4 bytes)
    int8_t qs[QK8_1]; // 8-bit signed quantized values (32 bytes)
} block_q8_1;

static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ============================================================================
// DP4A Helper
// ============================================================================

__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

// Load int32 from 2-byte aligned memory
__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

// Load int32 from 4-byte aligned memory
__device__ __forceinline__ int load_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// ============================================================================
// Helper: Read half from byte array
// ============================================================================

__device__ __forceinline__ float read_half(const uint8_t* ptr) {
    union { uint16_t u16; half f16; } un;
    un.u16 = *reinterpret_cast<const uint16_t*>(ptr);
    return __half2float(un.f16);
}

// ============================================================================
// Q4_1 × Q8_1 Dot Product
// ============================================================================

__device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const block_q4_1* __restrict__ bq4,
    const block_q8_1* __restrict__ bq8
) {
    int sumi = 0;

    // Q4_1 layout: qs[i] low nibble = x[i], high nibble = x[i+16]
    // Q8_1 layout: qs[i] = x[i] (sequential)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = load_int_b2(bq4->qs, i);
        int vi0 = (v >> 0) & 0x0F0F0F0F;  // x[i*4+0:3]
        int vi1 = (v >> 4) & 0x0F0F0F0F;  // x[i*4+16:19]

        int u0 = load_int_b4(bq8->qs, i);      // x[i*4+0:3]
        int u1 = load_int_b4(bq8->qs, i + 4);  // x[i*4+16:19]

        sumi = dp4a(vi0, u0, sumi);
        sumi = dp4a(vi1, u1, sumi);
    }

    // Extract Q4_1 parameters
    float d4 = read_half(reinterpret_cast<const uint8_t*>(&bq4->d));
    float m4 = read_half(reinterpret_cast<const uint8_t*>(&bq4->m));

    // Extract Q8_1 parameters (packed in half2)
    float d8 = read_half(reinterpret_cast<const uint8_t*>(&bq8->ds));
    float s8 = read_half(reinterpret_cast<const uint8_t*>(&bq8->ds) + 2);

    // Formula: d4*d8*sumi + m4*s8/4
    return d4 * d8 * sumi + m4 * s8 * 0.25f;
}

// ============================================================================
// GEMM Kernel: Q4_1 weight × FP32 activation (dynamically quantized to Q8_1)
// ============================================================================

__global__ void gemm_q4_1_fp32_kernel(
    const uint8_t* __restrict__ weight,  // Q4_1 quantized weight [N, K/32]
    const float* __restrict__ activation,  // FP32 activation [M, K]
    float* __restrict__ output,           // FP32 output [M, N]
    int M, int N, int K
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;  // Row in activation/batch
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // Column in output

    if (m >= M || n >= N) return;

    const int num_blocks = K / QK8_1;  // K must be divisible by 32

    // Pointer to this row's activation
    const float* act_row = activation + m * K;

    // Pointer to this column's weight blocks
    const block_q4_1* weight_col = reinterpret_cast<const block_q4_1*>(
        weight + n * num_blocks * sizeof(block_q4_1)
    );

    float sum = 0.0f;

    // Iterate over blocks (each block is 32 elements)
    for (int b = 0; b < num_blocks; b++) {
        // 1. Dynamically quantize 32 FP32 values to Q8_1
        block_q8_1 act_block;
        const int k_start = b * QK8_1;

        // Compute statistics for this block
        float act_max = 0.0f;
        float act_sum = 0.0f;
        float act_vals[QK8_1];

        #pragma unroll
        for (int i = 0; i < QK8_1; i++) {
            act_vals[i] = act_row[k_start + i];
            float abs_val = fabsf(act_vals[i]);
            if (abs_val > act_max) act_max = abs_val;
            act_sum += act_vals[i];
        }

        // Compute Q8_1 scale
        float d8 = act_max > 1e-6f ? act_max / 127.0f : 1.0f;

        // Store d8 and s8 in half2 format
        act_block.ds = __halves2half2(__float2half(d8), __float2half(act_sum));

        // Quantize to INT8
        #pragma unroll
        for (int i = 0; i < QK8_1; i++) {
            int q = (int)rintf(act_vals[i] / d8);
            act_block.qs[i] = (int8_t)max(-128, min(127, q));
        }

        // 2. Compute dot product with Q4_1 weight block
        sum += vec_dot_q4_1_q8_1(&weight_col[b], &act_block);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(weight.device()));

    // Use 16x16 thread blocks for simple implementation
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    gemm_q4_1_fp32_kernel<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 × FP32 Quantized GEMM");
}
