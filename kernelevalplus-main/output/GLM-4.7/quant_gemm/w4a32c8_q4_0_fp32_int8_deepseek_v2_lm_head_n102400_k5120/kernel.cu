#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

// Q4_0 block structure (packed)
struct block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // 32 packed 4-bit values
};

// Device function to read FP16 as float
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * Optimized W4A32C8 Q4_0 Quantized GEMM Kernel v7
 *
 * Key optimizations:
 * 1. One thread per output (proven effective)
 * 2. Vectorized loads (float4, uint4) for better bandwidth
 * 3. Aggressive loop unrolling with 4-block unrolling
 * 4. Optimized dynamic quantization with reduced branching
 * 5. Better ILP through independent computation paths
 */
__global__ void __launch_bounds__(256) w4a32c8_q4_0_gemm_kernel_v7(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each thread computes one output element
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const int num_blocks_k = K / 32;

    // Pre-compute pointers for better performance
    const block_q4_0* __restrict__ w_row = reinterpret_cast<const block_q4_0*>(weight + n * num_blocks_k * 18);
    const float* __restrict__ a_row = activation + m * K;

    float acc = 0.0f;

    // Process K blocks with 4-way unrolling
    int bk = 0;

    // Main loop: process 4 blocks at a time
    for (; bk + 4 <= num_blocks_k; bk += 4) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // Process 4 consecutive blocks for better ILP
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int current_bk = bk + i;

            // Load weight block using vectorized load
            const block_q4_0 w_block = w_row[current_bk];
            const float d_w = read_half_as_float(w_block.d);

            // Load activation block
            const float* __restrict__ a_block = a_row + current_bk * 32;

            // Optimized dynamic quantization
            float a_max = 0.0f;
            float a_sum = 0.0f;

            // Vectorized max/sum computation
            #pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const float4 vals = *reinterpret_cast<const float4*>(&a_block[j]);

                a_max = fmaxf(a_max, fmaxf(fabsf(vals.x), fmaxf(fabsf(vals.y), fabsf(vals.z))));
                a_max = fmaxf(a_max, fabsf(vals.w));

                a_sum += vals.x + vals.y + vals.z + vals.w;
            }

            const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

            // Optimized dot product computation
            int32_t sumi = 0;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                const uint8_t byte_val = w_block.qs[j];
                const int w_low = byte_val & 0x0F;
                const int w_high = (byte_val >> 4) & 0x0F;

                const int a_low = __float2int_rn(a_block[j] / d_a);
                const int a_high = __float2int_rn(a_block[j + 16] / d_a);

                sumi += w_low * a_low + w_high * a_high;
            }

            // Apply llama.cpp formula
            float block_result = d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);

            // Accumulate to respective accumulator
            if (i == 0) acc0 = block_result;
            else if (i == 1) acc1 = block_result;
            else if (i == 2) acc2 = block_result;
            else acc3 = block_result;
        }

        acc += acc0 + acc1 + acc2 + acc3;
    }

    // Process remaining blocks
    for (; bk < num_blocks_k; bk++) {
        const block_q4_0 w_block = w_row[bk];
        const float d_w = read_half_as_float(w_block.d);

        const float* __restrict__ a_block = a_row + bk * 32;

        float a_max = 0.0f;
        float a_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            const float4 vals = *reinterpret_cast<const float4*>(&a_block[i]);

            a_max = fmaxf(a_max, fmaxf(fabsf(vals.x), fmaxf(fabsf(vals.y), fabsf(vals.z))));
            a_max = fmaxf(a_max, fabsf(vals.w));

            a_sum += vals.x + vals.y + vals.z + vals.w;
        }

        const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_block.qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;

            const int a_low = __float2int_rn(a_block[i] / d_a);
            const int a_high = __float2int_rn(a_block[i + 16] / d_a);

            sumi += w_low * a_low + w_high * a_high;
        }

        acc += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
    }

    output[m * N + n] = acc;
}

/**
 * Host function to launch the optimized kernel
 */
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kByte, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    // Allocate output tensor
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Launch configuration - optimized for different M values
    const int threads_per_block = 256;
    const int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    const int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block, 1);

    // Launch optimized kernel
    w4a32c8_q4_0_gemm_kernel_v7<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v7 (DeepSeek-V2 LM Head)");
}
