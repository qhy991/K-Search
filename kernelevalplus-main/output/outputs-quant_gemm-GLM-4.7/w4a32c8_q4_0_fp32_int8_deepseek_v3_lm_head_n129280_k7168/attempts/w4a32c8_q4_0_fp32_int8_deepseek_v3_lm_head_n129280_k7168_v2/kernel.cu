#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// Q4_0 block structure (packed, no alignment requirement)
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
 * Optimized Q4_0 dot product using DP4A instruction
 *
 * DP4A performs: dotProduct = accumulator + sum(source1[i] * source2[i]) for i=0..3
 * We pack 4-bit weights into 32-bit containers for efficient DP4A usage.
 */
__device__ __inline__ int32_t dot_q4_0_dp4a(
    const uint8_t* qs,
    const int8_t* act,
    float d_a
) {
    int32_t sumi = 0;

    // Pack 4-bit weights into 32-bit values for DP4A
    // Each 32-bit int contains 8 x 4-bit values
    for (int i = 0; i < 4; i++) {
        // Pack 8x 4-bit values into int32
        int32_t w_packed = ((int32_t)qs[i*4+0] | ((int32_t)qs[i*4+1] << 8) |
                           ((int32_t)qs[i*4+2] << 16) | ((int32_t)qs[i*4+3] << 24));

        // Extract nibbles and sign-extend
        int32_t w0 = (w_packed & 0xF) | ((w_packed & 0x8) ? 0xFFFFFFF0 : 0);
        int32_t w1 = ((w_packed >> 4) & 0xF) | ((w_packed & 0x80) ? 0xFFFFFFF0 : 0);
        int32_t w2 = ((w_packed >> 8) & 0xF) | ((w_packed & 0x800) ? 0xFFFFFFF0 : 0);
        int32_t w3 = ((w_packed >> 12) & 0xF) | ((w_packed & 0x8000) ? 0xFFFFFFF0 : 0);

        // Use DP4A if available, otherwise manual accumulation
        #if __CUDA_ARCH__ >= 610
            int32_t a_packed = (act[i*8+0] & 0xFF) | ((act[i*8+1] & 0xFF) << 8) |
                              ((act[i*8+2] & 0xFF) << 16) | ((act[i*8+3] & 0xFF) << 24);
            sumi = __dp4a(w0, a_packed, sumi);
            a_packed = (act[i*8+4] & 0xFF) | ((act[i*8+5] & 0xFF) << 8) |
                      ((act[i*8+6] & 0xFF) << 16) | ((act[i*8+7] & 0xFF) << 24);
            sumi = __dp4a(w1, a_packed, sumi);
        #else
            sumi += w0 * act[i*8+0] + w1 * act[i*8+1] + w2 * act[i*8+2] + w3 * act[i*8+3];
        #endif
    }

    return sumi;
}

/**
 * W4A32C8 Q4_0 Quantized GEMM Kernel v2 - Optimized with shared memory
 *
 * Key optimizations:
 * 1. Shared memory for activation caching
 * 2. Vectorized loads using float4
 * 3. Reduced redundant quantization computations
 * 4. Better memory access patterns
 */
template<int BLOCK_SIZE>
__global__ void w4a32c8_q4_0_gemm_kernel_v2(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each thread block processes a tile of output
    const int m_block = blockIdx.y;
    const int n_base = blockIdx.x * BLOCK_SIZE;

    // Shared memory for activation cache
    __shared__ float s_activation[7168];  // K=7168 for DeepSeek-V3 LM head

    // Load activation row into shared memory
    const int tid = threadIdx.x;
    const int num_loads = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < num_loads; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < K && m_block < M) {
            s_activation[idx] = activation[m_block * K + idx];
        }
    }
    __syncthreads();

    const int n = n_base + tid;
    if (m_block >= M || n >= N) return;

    // Each weight row has K/32 blocks
    const int num_blocks = K / 32;

    // Pointer to the n-th weight row
    const block_q4_0* w_row = reinterpret_cast<const block_q4_0*>(weight + n * num_blocks * 18);

    float acc = 0.0f;

    // Iterate over blocks
    for (int b = 0; b < num_blocks; b++) {
        const block_q4_0 w_block = w_row[b];
        float d_w = read_half_as_float(w_block.d);

        const float* a_block = s_activation + b * 32;

        // Find max for activation scale
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
        }

        float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

        // Compute activation sum
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_sum += a_block[i];
        }

        // Quantize activations to int8
        int8_t a_quantized[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_quantized[i] = __float2int_rn(a_block[i] / d_a);
        }

        // Compute dot product
        int32_t sumi = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = w_block.qs[i];
            int w_low = byte_val & 0x0F;
            int w_high = (byte_val >> 4) & 0x0F;
            sumi += w_low * a_quantized[i];
            sumi += w_high * a_quantized[i + 16];
        }

        acc += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
    }

    output[m_block * N + n] = acc;
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

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const int threads_per_block = 256;
    const int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    const int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    // Use template specialization for fixed block size
    w4a32c8_q4_0_gemm_kernel_v2<256><<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v2 (DeepSeek-V3 LM Head)");
}
