/**
 * Q4_1 × FP32 GEMM Kernel
 *
 * Computation: C = A @ W^T
 * where A is FP32 activation [M, K]
 * and W is Q4_1 quantized weight [N, K/32]
 *
 * Q4_1 format per 32-element block:
 *   - d: half scale factor (2 bytes)
 *   - m: half minimum value (2 bytes)
 *   - qs[16]: packed 4-bit values (16 bytes, 2 values per byte)
 *
 * Dequantization: w_val = qs[i] * d + m
 * Nibble ordering: low nibbles (positions 0-15), then high nibbles (positions 16-31)
 *
 * This implementation uses the dequantization approach for FP32 activation:
 * Dynamically quantize activation to Q8_1 per block, then use the formula:
 *   result = d4 * d8 * sumi + m4 * s8 / 4
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Type Definitions
// ============================================================================

// Q4_1 block: 32 elements per block
typedef struct {
    half d;              // Scale factor (2 bytes)
    half m;              // Minimum value (2 bytes)
    uint8_t qs[16];      // Packed 4-bit values (16 bytes, 2 per byte)
} block_q4_1;

static_assert(sizeof(block_q4_1) == 20, "block_q4_1 must be 20 bytes");

// Q8_1 block: 32 elements per block
typedef struct {
    half2 ds;            // d (scale) and s (sum) packed as half2 (4 bytes)
    int8_t qs[32];       // 8-bit signed values (32 bytes)
} block_q8_1;

static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ============================================================================
// DP4A Instruction (Dot Product 4-element Accumulate)
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

// ============================================================================
// Helper Functions
// ============================================================================

// Load half as float (handles FP16 conversion correctly)
__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Load int32 from 2-byte aligned memory
__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = reinterpret_cast<const uint16_t*>(x);
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

// Load int32 from 4-byte aligned memory
__device__ __forceinline__ int load_int_b4(const void* x, int i32) {
    return reinterpret_cast<const int*>(x)[i32];
}

// ============================================================================
// Q4_1 × Q8_1 Dot Product
// ============================================================================

__device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const block_q4_1* __restrict__ bq4,
    const block_q8_1* __restrict__ bq8
) {
    int sumi = 0;

    // Q4_1 nibble layout: low nibbles first (positions 0-15), then high nibbles (positions 16-31)
    // This is llama.cpp compatible ordering
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = load_int_b2(bq4->qs, i);
        int vi0 = (v >> 0) & 0x0F0F0F0F;  // Low nibbles: positions [i*4, i*4+1, i*4+2, i*4+3]
        int vi1 = (v >> 4) & 0x0F0F0F0F;  // High nibbles: positions [16+i*4, 16+i*4+1, 16+i*4+2, 16+i*4+3]

        int u0 = load_int_b4(bq8->qs, i);      // Positions [i*4, i*4+1, i*4+2, i*4+3]
        int u1 = load_int_b4(bq8->qs, i + 4);  // Positions [16+i*4, 16+i*4+1, 16+i*4+2, 16+i*4+3]

        sumi = dp4a(vi0, u0, sumi);
        sumi = dp4a(vi1, u1, sumi);
    }

    // Extract parameters
    float d4 = half_to_float(*reinterpret_cast<const uint16_t*>(&bq4->d));
    float m4 = half_to_float(*reinterpret_cast<const uint16_t*>(&bq4->m));
    float d8 = __half2float(__low2half(bq8->ds));
    float s8 = __half2float(__high2half(bq8->ds));

    // Formula: d4 * d8 * sumi + m4 * s8 / 4
    return d4 * d8 * sumi + m4 * s8 / 4.0f;
}

// ============================================================================
// Dynamic Quantization: FP32 → Q8_1
// ============================================================================

__device__ __forceinline__ void quantize_fp32_to_q8_1(
    const float* __restrict__ fp32_vals,
    block_q8_1* __restrict__ q8_1_block
) {
    // Compute max absolute value for scaling
    float max_val = 0.0f;
    float sum_val = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float abs_val = fabsf(fp32_vals[i]);
        if (abs_val > max_val) max_val = abs_val;
        sum_val += fp32_vals[i];
    }

    // Compute scale and store sum
    float d8 = max_val / 127.0f;
    if (d8 < 1e-6f) d8 = 1e-6f;

    q8_1_block->ds = __halves2half2(__float2half(d8), __float2half(sum_val));

    // Quantize to INT8
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = static_cast<int>(roundf(fp32_vals[i] / d8));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        q8_1_block->qs[i] = static_cast<int8_t>(q);
    }
}

// ============================================================================
// Q4_1 × FP32 GEMM Kernel
// ============================================================================

/**
 * Each thread computes one output element C[m, n]
 *
 * Kernel strategy:
 * 1. Each thread loads its row of FP32 activation values
 * 2. Dynamically quantize activation to Q8_1 per block
 * 3. Load corresponding Q4_1 weight blocks
 * 4. Compute dot product using DP4A
 * 5. Apply compensation formula
 */
__global__ void q4_1_fp32_gemm_kernel(
    const block_q4_1* __restrict__ weight,    // [N, K/32]
    const float* __restrict__ activation,       // [M, K]
    float* __restrict__ output,                // [M, N]
    int M, int N, int K
) {
    // Thread ID maps to output element [m, n]
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) return;

    const int num_blocks = K / 32;
    float sum = 0.0f;

    // Pointer to activation row
    const float* act_row = activation + m * K;

    for (int b = 0; b < num_blocks; b++) {
        // Quantize 32 FP32 activation values to Q8_1
        block_q8_1 a_block;
        quantize_fp32_to_q8_1(act_row + b * 32, &a_block);

        // Get weight block at position (n, b)
        const block_q4_1* w_block = &weight[n * num_blocks + b];

        // Compute dot product with compensation
        sum += vec_dot_q4_1_q8_1(w_block, &a_block);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor forward(
    torch::Tensor weight,      // Q4_1 quantized weight [N, K/32]
    torch::Tensor activation,  // FP32 activation [M, K]
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(activation.is_contiguous(), "Activation must be contiguous");

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(weight.device()));

    // Launch kernel
    // Using 16x16 thread blocks for better occupancy
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    q4_1_fp32_gemm_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const block_q4_1*>(weight.data_ptr<uint8_t>()),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 x FP32 GEMM");
}
