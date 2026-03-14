#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Q4_1 block structure: 20 bytes per 32 elements
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

// Optimized kernel using warp-level primitives
__global__ void q4_1_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q4_1,
    const float* __restrict__ activation_fp32,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;
    const int m = blockIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n_base = blockIdx.y * blockDim.x + threadIdx.x;

    if (n_base >= N) return;

    const float* act_row = activation_fp32 + m * K;

    // Each thread processes its own N value
    float acc = 0.0f;

    // Process K blocks
    for (int kb = 0; kb < K_BLOCKS; kb++) {
        const block_q4_1* w_block = reinterpret_cast<const block_q4_1*>(
            weight_q4_1 + (n_base * K_BLOCKS + kb) * sizeof(block_q4_1)
        );

        float d_w = fp16_to_fp32(w_block->d);
        float m_w = fp16_to_fp32(w_block->m);

        int k_base = kb * 32;

        // Compute activation statistics
        float act_max_abs = 0.0f;
        float act_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float val = act_row[k_base + i];
            act_max_abs = fmaxf(act_max_abs, fabsf(val));
            act_sum += val;
        }
        float d_a = fmaxf(act_max_abs / 127.0f, 1e-6f);

        // Compute sumi
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_block->qs[i];
            int q_low = packed & 0x0F;
            int q_high = (packed >> 4) & 0x0F;

            int q8_low = __float2int_rn(act_row[k_base + i] / d_a);
            int q8_high = __float2int_rn(act_row[k_base + i + 16] / d_a);

            sumi += q_low * q8_low;
            sumi += q_high * q8_high;
        }

        acc += d_w * d_a * (float)sumi + m_w * act_sum;
    }

    output[m * N + n_base] = acc;
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

    // Optimize thread block size for RTX 4090
    const int THREADS_PER_BLOCK = 512;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(M, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

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
