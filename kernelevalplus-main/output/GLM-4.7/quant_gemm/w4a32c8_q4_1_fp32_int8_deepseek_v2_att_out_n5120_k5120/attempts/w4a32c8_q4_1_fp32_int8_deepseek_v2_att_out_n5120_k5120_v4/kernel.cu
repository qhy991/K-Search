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

// Optimized kernel with proper dp4a usage
__global__ void q4_1_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q4_1,
    const float* __restrict__ activation_fp32,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;
    const int m = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;

    if (n >= N) return;

    // Shared memory for pre-quantized activation
    __shared__ int8_t act_qs[160][32];  // [K_BLOCKS][32]
    __shared__ float act_d[160];  // scale per block
    __shared__ float act_s[160];  // sum per block

    // Each thread pre-quantizes its assigned activation blocks
    const float* act_row = activation_fp32 + m * K;
    for (int kb = threadIdx.x; kb < K_BLOCKS; kb += blockDim.x) {
        int k_base = kb * 32;

        // Compute statistics
        float act_max_abs = 0.0f;
        float act_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float val = act_row[k_base + i];
            act_max_abs = fmaxf(act_max_abs, fabsf(val));
            act_sum += val;
        }
        float d_a = fmaxf(act_max_abs / 127.0f, 1e-6f);

        act_d[kb] = d_a;
        act_s[kb] = act_sum;

        // Quantize
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            act_qs[kb][i] = (int8_t)__float2int_rn(act_row[k_base + i] / d_a);
        }
    }
    __syncthreads();

    float acc = 0.0f;

    // Process K blocks
    for (int kb = 0; kb < K_BLOCKS; kb++) {
        const block_q4_1* w_block = reinterpret_cast<const block_q4_1*>(
            weight_q4_1 + (n * K_BLOCKS + kb) * sizeof(block_q4_1)
        );

        float d_w = fp16_to_fp32(w_block->d);
        float m_w = fp16_to_fp32(w_block->m);
        float d_a = act_d[kb];
        float s_a = act_s[kb];

        const int8_t* __restrict__ a = act_qs[kb];
        const uint8_t* __restrict__ w_packed = w_block->qs;

        // Compute sumi = sum(q4_1[i] * q8_1[i])
        // Q4_1 values are in [0, 15], need to use them directly
        int sumi = 0;

        // Unroll 16 bytes = 32 Q4 values
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int q_low = w_packed[i] & 0x0F;
            int q_high = (w_packed[i] >> 4) & 0x0F;

            sumi += q_low * (int)a[i];
            sumi += q_high * (int)a[i + 16];
        }

        acc += d_w * d_a * (float)sumi + m_w * s_a;
    }

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
