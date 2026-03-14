#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 format: 18 bytes per block
// 2 bytes: FP16 scale
// 16 bytes: 32 packed 4-bit values (each byte contains 2 values)

// Simple correct kernel - based on v1 which passed correctness
__global__ void q4_0_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_blocks = K / 32;

    const int m_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const float* act_row = activation + m_idx * K;
    const uint8_t* w_ptr = weight_q + n_idx * K_blocks * 18;

    float sum = 0.0f;

    for (int block = 0; block < K_blocks; block++) {
        // Load scale (FP16) using aligned load at 2-byte boundary
        half2 scale_data = *reinterpret_cast<const half2*>(w_ptr + block * 18);
        float d_w = __half2float(scale_data.x);

        const uint8_t* qs_ptr = w_ptr + block * 18 + 2;
        const float* act_ptr = act_row + block * 32;

        // Unpack 4-bit values and compute
        for (int i = 0; i < 16; i++) {
            uint8_t byte = qs_ptr[i];
            int q_low = byte & 0x0F;
            int q_high = (byte >> 4) & 0x0F;

            float w_low = d_w * (float(q_low) - 8.0f);
            float w_high = d_w * (float(q_high) - 8.0f);

            sum += w_low * act_ptr[i];
            sum += w_high * act_ptr[i + 16];
        }
    }

    output[m_idx * N + n_idx] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_q = weight.data_ptr<uint8_t>();
    const float* activation_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Use optimal block sizes for RTX 4090
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    q4_0_fp32_gemm_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
