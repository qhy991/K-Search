#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Q4_0 block structure (18 bytes)
typedef struct {
    uint16_t d;        // scale (FP16)
    uint8_t qs[16];    // packed 4-bit quantized values
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "Q4_0 block size must be 18 bytes");

// Helper to read FP16 scale as float
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * Quantized GEMM Kernel: W4A32C8 (Q4_0 weights, FP32 activations, Q8_1 compute)
 *
 * Computes: output[m,n] = sum_{k}(weight[n,k] * activation[m,k])
 *
 * Where:
 * - weight is Q4_0 quantized (4-bit values with offset-8 encoding)
 * - activation is FP32, dynamically quantized to Q8_1 per block during compute
 * - computation uses 8-bit INT8 arithmetic
 *
 * Formula (from llama.cpp reference):
 *   result = d_w * (d_a * sumi - 8 * s_a)
 *
 * where:
 *   d_w = Q4_0 weight scale (FP16 -> FP32)
 *   d_a = Q8_1 activation scale (max/127.0)
 *   s_a = sum of original FP32 activation values in block
 *   sumi = sum of raw Q4_0 values * quantized Q8_1 activation values
 *
 * Q4_0 encoding:
 *   Each 16-byte block contains 32 4-bit values
 *   Packing: byte[i] = q[i] | (q[i+16] << 4)
 *   Decode: val = scale * (q - 8), where q in [0, 15]
 */
__global__ void quant_gemm_w4a32c8_kernel(
    const uint8_t* __restrict__ weight,      // Q4_0 weights: [N, K/32] blocks
    const float* __restrict__ activation,     // FP32 activation: [M, K]
    float* __restrict__ output,               // FP32 output: [M, N]
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;

    // Each thread computes one output element [m, n]
    // Grid-stride loop handles all M*N elements
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = global_tid; idx < M * N; idx += gridDim.x * blockDim.x) {
        int m = idx / N;
        int n = idx % N;

        const float* a_row = activation + m * K;
        const block_q4_0* w_row = reinterpret_cast<const block_q4_0*>(weight + n * K_BLOCKS * 18);

        float sum = 0.0f;

        // Iterate over K blocks (each block has 32 weights)
        for (int kb = 0; kb < K_BLOCKS; kb++) {
            // Q8_1 quantization of activation block:
            // 1. Compute scale (d_a) = max_abs / 127.0
            // 2. Compute sum (s_a) of original FP32 values
            // 3. Quantize each value: a_qs = round(a / d_a) in [-128, 127]
            // 4. Compute dot product with Q4_0 weights

            float act_sum = 0.0f;
            float act_max = 0.0f;
            int k_base = kb * 32;

            // Vectorized activation stats computation using float4
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                float4 a4 = *reinterpret_cast<const float4*>(a_row + k_base + i);
                float a0 = a4.x;
                float a1 = a4.y;
                float a2 = a4.z;
                float a3 = a4.w;

                act_sum += a0 + a1 + a2 + a3;

                a0 = fabsf(a0); a1 = fabsf(a1);
                a2 = fabsf(a2); a3 = fabsf(a3);
                float max4 = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
                act_max = fmaxf(act_max, max4);
            }

            // Q8_1 scale: d_a = max / 127.0
            float d_a = (act_max > 0.0f) ? act_max / 127.0f : 1.0f;
            float inv_d_a = 1.0f / d_a;

            // Get Q4_0 weight block
            float d_w = read_half_as_float(w_row[kb].d);
            const uint8_t* w_qs = w_row[kb].qs;

            // Compute dot product: sumi = sum(q_w_raw * a_qs)
            // IMPORTANT: Use raw Q4_0 values (0-15), NOT offset-corrected
            // The offset-8 correction is applied via the -8*s_a term in the final formula
            int sumi = 0;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int8_t q_low = static_cast<int8_t>(w_qs[i] & 0x0F);
                int8_t q_high = static_cast<int8_t>((w_qs[i] >> 4) & 0x0F);

                float a0 = a_row[k_base + i];
                float a1 = a_row[k_base + i + 16];

                sumi += q_low * static_cast<int8_t>(__float2int_rn(a0 * inv_d_a));
                sumi += q_high * static_cast<int8_t>(__float2int_rn(a1 * inv_d_a));
            }

            // Apply llama.cpp formula:
            // result += d_w * (d_a * sumi - 8 * s_a)
            sum += d_w * (d_a * static_cast<float>(sumi) - 8.0f * act_sum);
        }

        output[idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor weight,          // Q4_0 weights [N, K/32] as uint8 tensor
    torch::Tensor activation,       // FP32 activation [M, K]
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(activation.scalar_type() == torch::kFloat32, "Activation must be FP32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* activation_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch configuration
    // For large N (128256), using 256 threads per block
    // Grid size: enough blocks to cover all M*N elements
    int threads_per_block = 256;
    int num_blocks = (M * N + threads_per_block - 1) / threads_per_block;
    num_blocks = min(num_blocks, 128 * 256);  // Limit to reasonable grid size

    quant_gemm_w4a32c8_kernel<<<num_blocks, threads_per_block>>>(
        weight_ptr, activation_ptr, output_ptr, M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM W4A32C8");
}
