/**
 * W4A32C8 Q4_0 × FP32 GEMM for DeepSeek-V3 Attention Output
 *
 * Computes: C = A @ W^T where:
 * - A is FP32 activation [M, K]
 * - W is Q4_0 quantized weight [N, K/32]
 *
 * Q4_0 block format (18 bytes):
 * - d: FP16 scale (2 bytes)
 * - qs: 16 bytes of packed 4-bit values
 *
 * Q8_1 dynamic quantization on activation:
 * - Find max absolute value in each 32-element block
 * - Scale: d_a = max_abs / 127
 * - Sum: s_a = sum(activation)
 * - Quantized: q_a[i] = round(activation[i] / d_a)
 *
 * Formula: result = d4_0 * (d8_1 * sumi - 8 * s8_1)
 * where sumi = dot(q_a, q_w) using DP4A
 *
 * Dimensions: M (variable), N = 7168, K = 7168
 *
 * Optimization strategy for compute-bound kernel:
 * - 1D thread blocks: each thread computes one output element
 * - DP4A instruction for efficient 4-bit dot product
 * - Unrolled loops for better instruction-level parallelism
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

#define QK4_0 32
#define WARP_SIZE 32

// Q4_0 block structure (must match llama.cpp layout)
struct block_q4_0 {
    uint16_t d;          // FP16 scale/dequantization factor
    uint8_t qs[16];      // packed 4-bit values (32 values)
};

// Convert uint16 to float (FP16)
inline __device__ float half_to_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

// DP4A intrinsic (available on CC >= 6.1)
#if __CUDA_ARCH__ >= 610
inline __device__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
inline __device__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

/**
 * Process one Q4_0 block with Q8_1 dynamic quantization
 * Uses the formula: result = d_w * (d_a * sumi - 8 * s_a)
 */
inline __device__ float process_q4_0_block(
    const block_q4_0* w_block,
    const float* act
) {
    // Extract weight scale
    const float d_w = half_to_float(w_block->d);

    // Q8_1 dynamic quantization: find scale and sum for activation
    float a_max = 0.0f;
    float s_a = 0.0f;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float4 v = *reinterpret_cast<const float4*>(&act[j * 4]);
        a_max = fmaxf(a_max, fabsf(v.x));
        a_max = fmaxf(a_max, fabsf(v.y));
        a_max = fmaxf(a_max, fabsf(v.z));
        a_max = fmaxf(a_max, fabsf(v.w));
        s_a += v.x + v.y + v.z + v.w;
    }

    const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
    const float inv_d_a = 127.0f / a_max;

    // Compute dot product using DP4A
    // Unpack Q4_0 in llama.cpp format: first all low nibbles, then all high nibbles
    int sumi = 0;
    const uint8_t* qs = w_block->qs;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Pack 4 Q4_0 low nibbles into 32-bit
        const int w_low = (qs[j * 4 + 0] & 0x0F) |
                         ((qs[j * 4 + 1] & 0x0F) << 8) |
                         ((qs[j * 4 + 2] & 0x0F) << 16) |
                         ((qs[j * 4 + 3] & 0x0F) << 24);

        // Pack 4 Q4_0 high nibbles into 32-bit
        const int w_high = (qs[j * 4 + 0] >> 4) |
                          ((qs[j * 4 + 1] >> 4) << 8) |
                          ((qs[j * 4 + 2] >> 4) << 16) |
                          ((qs[j * 4 + 3] >> 4) << 24);

        // Quantize activation (low nibbles: positions 0-15)
        const int a_low = ((int)(uint8_t)__float2int_rn(act[j * 4 + 0] * inv_d_a)) |
                         (((int)(uint8_t)__float2int_rn(act[j * 4 + 1] * inv_d_a)) << 8) |
                         (((int)(uint8_t)__float2int_rn(act[j * 4 + 2] * inv_d_a)) << 16) |
                         (((int)(uint8_t)__float2int_rn(act[j * 4 + 3] * inv_d_a)) << 24);

        // Quantize activation (high nibbles: positions 16-31)
        const int a_high = ((int)(uint8_t)__float2int_rn(act[j * 4 + 16] * inv_d_a)) |
                          (((int)(uint8_t)__float2int_rn(act[j * 4 + 17] * inv_d_a)) << 8) |
                          (((int)(uint8_t)__float2int_rn(act[j * 4 + 18] * inv_d_a)) << 16) |
                          (((int)(uint8_t)__float2int_rn(act[j * 4 + 19] * inv_d_a)) << 24);

        sumi = dp4a(a_low, w_low, sumi);
        sumi = dp4a(a_high, w_high, sumi);
    }

    // Apply the formula: d_w * (d_a * sumi - 8 * s_a)
    return d_w * (d_a * (float)sumi - 8.0f * s_a);
}

/**
 * Q4_0 GEMM kernel - 1D thread per output element
 * Each thread computes one output element C[row, col]
 */
__global__ void q4_0_gemm_kernel(
    const float* __restrict__ activation,
    const block_q4_0* __restrict__ weight,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Compute output position
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (row >= M || col >= N) return;

    const float* a_row = activation + row * K;
    const block_q4_0* w_col = weight + col * (K / QK4_0);

    float sum = 0.0f;
    const int num_blocks = K / QK4_0;

    // Process each Q4_0 block
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        sum += process_q4_0_block(&w_col[block_idx], a_row + block_idx * QK4_0);
    }

    output[row * N + col] = sum;
}

/**
 * Host function to launch the Q4_0 GEMM kernel
 */
torch::Tensor forward(
    torch::Tensor weight,      // [N, K/32 * 18] Q4_0 blocks (as uint8 bytes)
    torch::Tensor activation,  // [M, K] FP32
    int M, int N, int K
) {
    // Validate dimensions
    assert(K == 7168);
    assert(N == 7168);
    assert(K % QK4_0 == 0);

    // Create output tensor
    auto output = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    // Cast weight to block_q4_0 pointer
    const block_q4_0* weight_ptr = reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>());

    // Configure kernel launch parameters
    // Use 2D thread blocks: each thread computes one output element
    dim3 blockDim(32, 8);  // 32 threads in N dim, 8 in M dim
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    q4_0_gemm_kernel<<<gridDim, blockDim>>>(
        activation.data_ptr<float>(),
        weight_ptr,
        output.data_ptr<float>(),
        M, N, K
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 Quantized GEMM for DeepSeek-V3 Attention Output");
}
