#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Q4_1 block structure: 20 bytes per 32 elements
// Matches GGML/llama.cpp format
typedef struct {
    uint16_t d;  // FP16 scale
    uint16_t m;  // FP16 min
    uint8_t qs[16];  // packed 4-bit values
} block_q4_1;
static_assert(sizeof(block_q4_1) == 20, "Q4_1 block size must be 20 bytes");

// Device function to convert FP16 to FP32
__device__ __inline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Core kernel for W4A32C8 Q4_1 × FP32 GEMM with dynamic Q8_1 quantization
// Formula: output = d4_1 * d8_1 * sumi + m4_1 * s8_1
__global__ void q4_1_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q4_1,  // [N, K/32] in Q4_1 blocks
    const float* __restrict__ activation_fp32,  // [M, K]
    float* __restrict__ output,  // [M, N]
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;  // Number of 32-element blocks in K dimension

    // Each thread block processes one row of output (one M value)
    const int m = blockIdx.x;

    // Use shared memory for the activation row
    __shared__ float act_shared[32];

    // Pointer to the activation row for this M
    const float* act_row = activation_fp32 + m * K;

    // Each thread processes one N value
    const int n = blockIdx.y * blockDim.x + threadIdx.x;

    if (n >= N) return;

    float acc = 0.0f;

    // Loop over K dimension in blocks of 32
    for (int kb = 0; kb < K_BLOCKS; kb++) {
        // Load activation block into shared memory (32 elements)
        if (threadIdx.x < 32) {
            int k_idx = kb * 32 + threadIdx.x;
            act_shared[threadIdx.x] = act_row[k_idx];
        }
        __syncthreads();

        // Load weight block for this N and this K block
        // Weight layout: [N, K_BLOCKS] of block_q4_1
        const block_q4_1* w_block = reinterpret_cast<const block_q4_1*>(
            weight_q4_1 + (n * K_BLOCKS + kb) * sizeof(block_q4_1)
        );

        // Decode Q4_1 weight scale and min
        float d_w = fp16_to_fp32(w_block->d);
        float m_w = fp16_to_fp32(w_block->m);

        // Compute activation statistics for Q8_1 dynamic quantization
        // d_a = max_abs(act) / 127, s_a = sum(act)
        float act_max_abs = 0.0f;
        float act_sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            float val = act_shared[i];
            float abs_val = fabsf(val);
            if (abs_val > act_max_abs) act_max_abs = abs_val;
            act_sum += val;
        }
        float d_a = fmaxf(act_max_abs / 127.0f, 1e-6f);

        // Compute sumi = sum(q4_1[i] * q8_1[i])
        // where q8_1[i] = round(act[i] / d_a)
        int sumi = 0;

        // Unroll the packed Q4_1 values
        // Each byte contains 2 Q4 values (low nibble = value 0-15, high nibble = value 16-31)
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_block->qs[i];
            int q_low = packed & 0x0F;           // Value 0-15
            int q_high = (packed >> 4) & 0x0F;   // Value 16-31

            float act_low = act_shared[i];
            float act_high = act_shared[i + 16];

            // Quantize activation to INT8
            int q8_low = __float2int_rn(act_low / d_a);
            int q8_high = __float2int_rn(act_high / d_a);

            // Accumulate dot product
            sumi += q_low * q8_low;
            sumi += q_high * q8_high;
        }

        // Apply W4A32C8 formula: d_w * d_a * sumi + m_w * s_a
        acc += d_w * d_a * (float)sumi + m_w * act_sum;

        __syncthreads();
    }

    // Write output
    output[m * N + n] = acc;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.scalar_type() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Launch kernel with 2D grid
    // blockIdx.x = M (batch dimension)
    // blockIdx.y = N / TILE_N (output feature dimension)
    const int TILE_N = 256;
    dim3 block(TILE_N);
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);

    q4_1_fp32_gemm_kernel<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_1 FP32 GEMM with dynamic Q8_1 activation quantization");
}
