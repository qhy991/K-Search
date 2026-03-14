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
 * Optimized W4A32C8 Q4_0 Quantized GEMM Kernel v6
 *
 * Key optimizations:
 * 1. Shared memory for activation quantization (reused across N)
 * 2. Vectorized loads (float4) for activations
 * 3. Better ILP with pipelined computation
 */
__global__ void __launch_bounds__(256) w4a32c8_q4_0_gemm_kernel_v6(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K,
    int block_size_m  // Runtime parameter for block size
) {
    // Shared memory for quantized activations (reused across N)
    __shared__ float s_a_block[32];         // Activation values (FP32)
    __shared__ float s_d_a;                // Activation scale
    __shared__ float s_a_sum;              // Activation sum

    const int num_blocks_k = K / 32;

    // Block and thread indices
    const int m_base = blockIdx.x * block_size_m;
    const int n_base = blockIdx.y * 32;  // Fixed N block size of 32

    const int m_local = threadIdx.x / 32;
    const int n_local = threadIdx.x % 32;

    const int m = m_base + m_local;
    const int n = n_base + n_local;

    // Accumulator
    float acc = 0.0f;

    // Only proceed if in range
    const bool valid_m = (m < M) && (m_local < block_size_m);
    const bool valid_n = (n < N);

    // Pre-compute pointers
    const float* a_row = activation + (valid_m ? m : 0) * K;

    // Process K blocks
    for (int bk = 0; bk < num_blocks_k; bk++) {
        // Load and quantize activation block (done once per M)
        if (m_local == 0 && valid_m) {
            const float* a_block = a_row + bk * 32;

            // Vectorized load using float4
            float a_max = 0.0f;
            float a_sum = 0.0f;

            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                float4 vals = *reinterpret_cast<const float4*>(&a_block[i]);
                s_a_block[i] = vals.x;
                s_a_block[i + 1] = vals.y;
                s_a_block[i + 2] = vals.z;
                s_a_block[i + 3] = vals.w;

                float m0 = fabsf(vals.x);
                float m1 = fabsf(vals.y);
                float m2 = fabsf(vals.z);
                float m3 = fabsf(vals.w);

                a_max = fmaxf(a_max, fmaxf(fmaxf(m0, m1), fmaxf(m2, m3)));
                a_sum += vals.x + vals.y + vals.z + vals.w;
            }

            s_d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;
            s_a_sum = a_sum;
        }

        __syncthreads();

        if (valid_m && valid_n) {
            // Load weight block
            const block_q4_0* w_block = reinterpret_cast<const block_q4_0*>(
                weight + n * num_blocks_k * 18 + bk * 18
            );
            const float d_w = read_half_as_float(w_block->d);

            // Compute dot product
            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const uint8_t byte_val = w_block->qs[i];
                const int w_low = byte_val & 0x0F;
                const int w_high = (byte_val >> 4) & 0x0F;

                const int a_low = __float2int_rn(s_a_block[i] / s_d_a);
                const int a_high = __float2int_rn(s_a_block[i + 16] / s_d_a);

                sumi += w_low * a_low + w_high * a_high;
            }

            // Apply formula
            acc += d_w * (s_d_a * static_cast<float>(sumi) - 8.0f * s_a_sum);
        }

        __syncthreads();
    }

    // Write result
    if (valid_m && valid_n) {
        output[m * N + n] = acc;
    }
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

    // Adaptive configuration based on M
    int block_size_m;
    if (M <= 2) block_size_m = M;
    else if (M <= 8) block_size_m = 2;
    else if (M <= 32) block_size_m = 4;
    else block_size_m = 8;

    const int blocks_x = (M + block_size_m - 1) / block_size_m;
    const int blocks_y = (N + 32 - 1) / 32;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(block_size_m * 32);

    // Launch optimized kernel
    w4a32c8_q4_0_gemm_kernel_v6<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K,
        block_size_m
    );

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v6 (DeepSeek-V2 LM Head)");
}
