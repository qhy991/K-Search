#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

// Helper: Convert FP16 to FP32
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Q4_0 × Q8_1 style dot product
// This follows the llama.cpp pattern:
// - Dynamically quantize activation to INT8 per 32-element block
// - Compute INT8 dot product with Q4_0 weights
// - Apply scales: result = d4_0 * (d8_1 * sumi - 8 * s8_1)
//
// The key insight: using INT8 accumulation allows us to use dp4a instructions
// and only apply FP32 scales at the end.
__device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const float* __restrict__ act,
    const uint8_t* __restrict__ wq,
    int K
) {
    // Q4_0 format: 34 bytes per 32 values
    // - 2 bytes: scale (FP16)
    // - 16 bytes: packed 4-bit values (32 values stored as 16 bytes)
    // - 16 bytes: padding (to make 34 bytes total, but we only use 18 bytes)

    // However, looking at the quantization, we have 18 bytes per block
    // - 2 bytes: scale (FP16)
    // - 16 bytes: packed 4-bit values

    const int num_blocks = K / 32;

    // Accumulators for the llama.cpp formula
    // result = d4_0 * (d8_1 * sumi - 8 * s8_1)
    // where sumi is the INT8 dot product
    // d8_1 is the activation scale per block
    // s8_1 is the activation sum per block

    float result = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        // === Load Q4_0 weight block ===
        // Scale is at offset 0
        uint16_t w_scale16;
        memcpy(&w_scale16, &wq[b * 18], 2);
        const float d4_0 = fp16_to_fp32(w_scale16);

        // Packed 4-bit values start at offset 2
        const uint8_t* w_packed = &wq[b * 18 + 2];

        // === Quantize activation block to Q8_1 style ===
        // Find max absolute value for scaling
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(act[i]));
        }

        // Scale for activation: d8_1 = max / 127
        const float d8_1 = a_max > 0 ? a_max / 127.0f : 1.0f;

        // Compute sum of activation values (s8_1 in llama.cpp)
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_sum += act[i];
        }

        // === INT8 dot product ===
        // Each Q4_0 byte contains two 4-bit values
        // Low nibble = first value, high nibble = second value
        // Q4_0 encoding: values 0-15 represent -8 to +7
        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            // Load packed byte
            uint8_t packed = w_packed[i];

            // Extract low and high nibbles (0-15 range)
            int w_low = packed & 0x0F;
            int w_high = (packed >> 4) & 0x0F;

            // Quantize activation values to INT8
            float a0 = act[i];
            float a1 = act[i + 16];
            int8_t aq0 = __float2int_rn(fmaxf(-128.0f, fminf(127.0f, a0 / d8_1)));
            int8_t aq1 = __float2int_rn(fmaxf(-128.0f, fminf(127.0f, a1 / d8_1)));

            // Accumulate dot product
            sumi += w_low * aq0;
            sumi += w_high * aq1;
        }

        // Apply scales (llama.cpp formula)
        // result += d4_0 * (d8_1 * sumi - 8 * a_sum)
        result += d4_0 * (d8_1 * __int2float_rn(sumi) - 8.0f * a_sum);

        // Advance to next block
        act += 32;
    }

    return result;
}

// Strategy 1: Small M (M=1) - One thread per output element
__global__ __launch_bounds__(1024, 1)
void gemm_q4_0_small_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    if (m >= M) return;

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];

    output[m * N + n] = vec_dot_q4_0_q8_1(act, w, K);
}

// Strategy 2: Medium M - 2D tiling
__global__ void gemm_q4_0_medium_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int bytes_per_row = (K / 32) * 18;
    const float* act = &activation[m * K];
    const uint8_t* w = &weight[n * bytes_per_row];

    output[m * N + n] = vec_dot_q4_0_q8_1(act, w, K);
}

torch::Tensor forward(
    torch::Tensor weight_q4,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be FP32");
    TORCH_CHECK(activation.device().is_cuda(), "Activation must be on CUDA device");
    TORCH_CHECK(weight_q4.device().is_cuda(), "Weight must be on CUDA device");

    auto options = torch::dtype(torch::kFloat32)
                      .device(activation.device())
                      .layout(torch::kStrided);

    auto output = torch::empty({M, N}, options);

    auto w_ptr = weight_q4.data_ptr<uint8_t>();
    auto a_ptr = activation.data_ptr<float>();
    auto o_ptr = output.data_ptr<float>();

    // Strategy dispatch based on batch size M
    if (M == 1) {
        // Strategy 1: Single token - max threads per row
        const int threads = 1024;
        const dim3 blocks((N + threads - 1) / threads, M);
        gemm_q4_0_small_m<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else if (M <= 8) {
        // Strategy 1: Small batch - use 1D threads per row
        const int threads = 512;
        const dim3 blocks((N + threads - 1) / threads, M);
        gemm_q4_0_small_m<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    } else {
        // Strategy 2: Larger batch - 2D tiling
        const int TILE_N = 128;
        const int TILE_M = 8;
        const dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        const dim3 threads(TILE_N, TILE_M);
        gemm_q4_0_medium_m<<<blocks, threads>>>(w_ptr, a_ptr, o_ptr, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM - Q4_0×Q8_1 pattern");
}
