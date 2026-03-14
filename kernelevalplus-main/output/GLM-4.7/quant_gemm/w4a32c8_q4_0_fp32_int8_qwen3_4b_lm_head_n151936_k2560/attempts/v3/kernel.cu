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
 * W4A32C8 Q4_0 Quantized GEMM Kernel v3 - Optimized
 *
 * Key optimizations over v1:
 * 1. Reduced branching in dot product loop
 * 2. Fused activation statistics computation with quantization
 * 3. Better register usage
 * 4. Improved memory access patterns
 */
__global__ void __launch_bounds__(256) w4a32c8_q4_0_gemm_kernel_v3(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const int num_blocks_k = K / 32;

    // Pre-compute pointers
    const block_q4_0* __restrict__ w_row = reinterpret_cast<const block_q4_0*>(weight + n * num_blocks_k * 18);
    const float* __restrict__ a_row = activation + m * K;

    float acc = 0.0f;

    // Process K blocks with 4-way unrolling
    int bk = 0;

    for (; bk + 4 <= num_blocks_k; bk += 4) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int current_bk = bk + i;

            const block_q4_0 w_block = w_row[current_bk];
            const float d_w = read_half_as_float(w_block.d);

            const float* __restrict__ a_block = a_row + current_bk * 32;

            // Optimized activation statistics: combine max, sum, and quantization
            float a_max = 0.0f;
            float a_sum = 0.0f;

            // First pass: find max and compute sum
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                const float val = a_block[j];
                const float abs_val = fabsf(val);
                if (abs_val > a_max) a_max = abs_val;
                a_sum += val;
            }

            const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;
            const float inv_d_a = 1.0f / d_a;

            // Dot product with inline quantization
            int32_t sumi = 0;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                const uint8_t byte_val = w_block.qs[j];
                const int w_low = byte_val & 0x0F;
                const int w_high = (byte_val >> 4) & 0x0F;

                const int a_low = __float2int_rn(a_block[j] * inv_d_a);
                const int a_high = __float2int_rn(a_block[j + 16] * inv_d_a);

                sumi += w_low * a_low + w_high * a_high;
            }

            const float block_result = d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);

            if (i == 0) acc0 = block_result;
            else if (i == 1) acc1 = block_result;
            else if (i == 2) acc2 = block_result;
            else acc3 = block_result;
        }

        acc += acc0 + acc1 + acc2 + acc3;
    }

    for (; bk < num_blocks_k; bk++) {
        const block_q4_0 w_block = w_row[bk];
        const float d_w = read_half_as_float(w_block.d);

        const float* __restrict__ a_block = a_row + bk * 32;

        float a_max = 0.0f;
        float a_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            const float val = a_block[i];
            const float abs_val = fabsf(val);
            if (abs_val > a_max) a_max = abs_val;
            a_sum += val;
        }

        const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_block.qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;

            const int a_low = __float2int_rn(a_block[i] * inv_d_a);
            const int a_high = __float2int_rn(a_block[i + 16] * inv_d_a);

            sumi += w_low * a_low + w_high * a_high;
        }

        acc += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
    }

    output[m * N + n] = acc;
}

/**
 * Host function to launch the kernel
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

    // Launch configuration
    const int threads_per_block = 256;
    const int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    const int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block, 1);

    // Launch kernel
    w4a32c8_q4_0_gemm_kernel_v3<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v3 (Qwen3-4B LM Head) - Optimized");
}
