#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Q4_1 Quantized GEMM Kernel V7 (W4A32C8) - Final Correct Version
 *
 * Correct implementation based on v5 with minor optimizations
 */

__device__ __inline__ float fp16_to_fp32(uint16_t x) {
    uint32_t exp = ((x & 0x7C00) >> 10) - 15 + 127;
    uint32_t mant = x & 0x03FF;
    uint32_t result = ((x & 0x8000) << 16) | (exp << 23) | (mant << 13);
    return __int_as_float(result);
}

__global__ void q4_1_gemm_kernel(
    const uint8_t* __restrict__ weight_q4_1,
    const float* __restrict__ activation_fp32,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    const float* act_row = activation_fp32 + m * K;
    float sum = 0.0f;

    for (int kb = 0; kb < K_BLOCKS; kb++) {
        const float* act_block = act_row + kb * 32;

        float act_max = 0.0f;
        float act_sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            float val = act_block[i];
            act_max = fmaxf(act_max, fabsf(val));
            act_sum += val;
        }
        float d_a = fmaxf(act_max / 127.0f, 1e-6f);

        const uint8_t* w_block = weight_q4_1 + (n * K_BLOCKS + kb) * 20;
        float d_w = fp16_to_fp32(*(uint16_t*)(w_block + 0));
        float m_w = fp16_to_fp32(*(uint16_t*)(w_block + 2));
        const uint8_t* w_packed = w_block + 4;

        int32_t sumi = 0;
        for (int i = 0; i < 16; i++) {
            int a_q0 = __float2int_rn(act_block[i] / d_a);
            int a_q1 = __float2int_rn(act_block[i + 16] / d_a);
            a_q0 = max(-128, min(127, a_q0));
            a_q1 = max(-128, min(127, a_q1));
            sumi += (int)(w_packed[i] & 0x0F) * a_q0;
            sumi += (int)((w_packed[i] >> 4) & 0x0F) * a_q1;
        }

        sum += d_w * d_a * (float)sumi + m_w * act_sum;
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight_q4_1, torch::Tensor activation_fp32, int M, int N, int K) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation_fp32.device()));

    const uint8_t* weight_ptr = weight_q4_1.data_ptr<uint8_t>();
    const float* activation_ptr = activation_fp32.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    const int THREADS = 256;
    dim3 block(THREADS);
    dim3 grid((N + THREADS - 1) / THREADS, M);

    q4_1_gemm_kernel<<<grid, block>>>(weight_ptr, activation_ptr, output_ptr, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 W4A32C8 Quantized GEMM forward V7");
}
