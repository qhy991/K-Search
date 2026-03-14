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
 * Optimized W4A32C8 Q4_0 Quantized GEMM Kernel v2
 *
 * Key optimizations:
 * 1. Shared memory for weight blocks (reuse across M dimension)
 * 2. Vectorized loads (float4) for activations
 * 3. Coalesced global memory access
 * 4. Reduced register pressure
 * 5. Better thread block configuration
 */
template <int BLOCK_SIZE>
__global__ void w4a32c8_q4_0_gemm_kernel_v2(
    const uint8_t* __restrict__ weight,      // Q4_0 weights [N, K/32, 18]
    const float* __restrict__ activation,    // FP32 activations [M, K]
    float* __restrict__ output,              // FP32 output [M, N]
    int M, int N, int K
) {
    // Shared memory for weight block cache
    __shared__ float s_d_w[BLOCK_SIZE];  // Weight scales
    __shared__ uint8_t s_qs[BLOCK_SIZE][16];  // Weight quantized values

    // Each thread block processes a block of N outputs
    const int n_base = blockIdx.x * BLOCK_SIZE;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int num_blocks_k = K / 32;

    // Each thread in the block handles one N dimension
    const int n_local = threadIdx.x;
    const int n = n_base + n_local;

    // Accumulator
    float acc = 0.0f;

    // Pointer to the m-th activation row
    const float* a_row = activation + m * K;

    // Iterate over K blocks
    for (int bk = 0; bk < num_blocks_k; bk++) {
        // Load weight block into shared memory
        // All threads cooperatively load BLOCK_SIZE weight blocks
        #pragma unroll
        for (int i = 0; i < (BLOCK_SIZE + 31) / 32; i++) {
            int load_idx = threadIdx.x + i * 32;
            if (load_idx < BLOCK_SIZE && (n_base + load_idx) < N) {
                const block_q4_0* w_block_ptr = reinterpret_cast<const block_q4_0*>(
                    weight + (n_base + load_idx) * num_blocks_k * 18 + bk * 18
                );
                const block_q4_0 w_block = *w_block_ptr;
                s_d_w[load_idx] = read_half_as_float(w_block.d);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    s_qs[load_idx][j] = w_block.qs[j];
                }
            }
        }

        __syncthreads();

        if (n < N) {
            // Load activation block (32 values) using vectorized loads
            const float* a_block = a_row + bk * 32;

            // Dynamically quantize activation to Q8_1 style
            float a_max = 0.0f;
            float a_sum = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 vals = *reinterpret_cast<const float4*>(&a_block[i * 4]);
                a_max = fmaxf(a_max, fmaxf(fmaxf(fabsf(vals.x), fabsf(vals.y)), fmaxf(fabsf(vals.z), fabsf(vals.w))));
                a_sum += vals.x + vals.y + vals.z + vals.w;
            }

            float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

            // Compute dot product using cached weight data
            float d_w = s_d_w[n_local];

            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t byte_val = s_qs[n_local][i];
                int w_low = byte_val & 0x0F;
                int w_high = (byte_val >> 4) & 0x0F;

                float a0 = a_block[i];
                float a1 = a_block[i + 16];

                int a_low = __float2int_rn(a0 / d_a);
                int a_high = __float2int_rn(a1 / d_a);

                sumi += w_low * a_low;
                sumi += w_high * a_high;
            }

            acc += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
        }

        __syncthreads();
    }

    if (n < N) {
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

    // Launch configuration optimized for shared memory usage
    // Block size for N dimension (tradeoff between shared memory and parallelism)
    const int BLOCK_SIZE_N = 64;  // Each block processes 64 output elements

    const int blocks_x = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    const int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(BLOCK_SIZE_N);

    // Launch optimized kernel
    w4a32c8_q4_0_gemm_kernel_v2<BLOCK_SIZE_N><<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v2 (DeepSeek-V2 LM Head)");
}
