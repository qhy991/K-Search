#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_1 block structure (20 bytes per block of 32 values)
// Layout: scale (FP16, 2 bytes) + min (FP16, 2 bytes) + qs (uint4[32] packed into 16 bytes)
typedef struct {
    uint16_t d;      // scale (FP16)
    uint16_t m;      // min (FP16)
    uint8_t qs[16];  // packed 4-bit values (32 values, 8 per byte)
} block_q4_1;
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 must be 20 bytes");

// Helper to read FP16 from uint16
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Helper to compute sum of products with 4-bit to 8-bit signed conversion
// Q4_1 values are stored as unsigned 0-15 but represent signed -8 to 7
__device__ __inline__ int dot_product_q4_1_q8_1(const uint8_t* w_qs, const int8_t* a_qs) {
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // Extract two 4-bit values from w_qs[i]
        // Low nibble: value (0-15) -> signed (-8 to 7) by subtracting 8
        int w_low = (w_qs[i] & 0x0F) - 8;
        // High nibble: value (0-15) -> signed (-8 to 7) by subtracting 8
        int w_high = ((w_qs[i] >> 4) & 0x0F) - 8;
        sum += w_low * a_qs[i];
        sum += w_high * a_qs[i + 16];
    }
    return sum;
}

// Main kernel: Q4_1 weight matrix multiply with FP32 activation
// Output shape: [M, N]
// Weight shape: [N, K/32] in Q4_1 format
// Activation shape: [M, K] in FP32
__global__ void q4_1_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation_fp32,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each thread block processes one output row (M dimension)
    const int m = blockIdx.x;
    if (m >= M) return;

    // Each thread in block handles one output column (N dimension)
    const int n = threadIdx.x;
    const int n_stride = blockDim.x;

    const float* a_row = activation_fp32 + m * K;
    const int num_blocks = K / 32;

    // Accumulator for this thread's output element(s)
    float sum = 0.0f;

    // Process all blocks (K dimension)
    for (int b = 0; b < num_blocks; b++) {
        // Dynamically quantize this block of the activation row to Q8_1
        const float* a_block_ptr = a_row + b * 32;

        // Compute statistics for Q8_1 quantization
        float a_max = 0.0f;
        float a_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float val = a_block_ptr[i];
            float abs_val = fabsf(val);
            if (abs_val > a_max) a_max = abs_val;
            a_sum += val;
        }

        // Q8_1 scale (d_a)
        float d_a = a_max / 127.0f;
        if (d_a < 1e-6f) d_a = 1e-6f;

        // Quantize activation to int8
        int8_t a_qs[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int q = __float2int_rn(a_block_ptr[i] / d_a);
            a_qs[i] = (int8_t)max(-128, min(127, q));
        }

        // Process weight blocks for this output column
        for (int n_base = n; n_base < N; n_base += n_stride) {
            // Load Q4_1 weight block
            const block_q4_1* w_block = reinterpret_cast<const block_q4_1*>(
                weight_q + (n_base * num_blocks + b) * 20
            );

            // Read weight scale and min
            float d_w = read_half_as_float(w_block->d);
            float m_w = read_half_as_float(w_block->m);

            // Compute dot product: sum((w_qs - 8) * a_qs)
            int sumi = dot_product_q4_1_q8_1(w_block->qs, a_qs);

            // Apply formula: result = d_w * d_a * sumi + m_w * s_a
            // Note: w_qs is treated as signed by subtracting 8 in the dot product
            float result = d_w * d_a * (float)sumi + m_w * a_sum;
            sum += result;
        }
    }

    // Write output
    for (int n_base = n; n_base < N; n_base += n_stride) {
        output[m * N + n_base] = sum;
    }
}

// Host wrapper function
torch::Tensor forward(
    torch::Tensor weight_q,
    torch::Tensor activation_fp32,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight_q.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation_fp32.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight_q.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(activation_fp32.is_contiguous(), "Activation must be contiguous");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight_q.device()));

    // Launch configuration
    const int threads_per_block = min(256, (int)N);
    const int blocks_per_grid = M;

    q4_1_fp32_gemm_kernel<<<blocks_per_grid, threads_per_block>>>(
        weight_q.data_ptr<uint8_t>(),
        activation_fp32.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 FP32 GEMM");
}
