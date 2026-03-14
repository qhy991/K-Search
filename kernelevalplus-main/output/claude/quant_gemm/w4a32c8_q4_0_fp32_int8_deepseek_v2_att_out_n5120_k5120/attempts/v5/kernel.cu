#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 format: 18 bytes per block
// 2 bytes: FP16 scale
// 16 bytes: 32 packed 4-bit values (each byte contains 2 values)
// Packing: byte[i] = q[i] | (q[i+16] << 4)

// Optimized kernel that processes multiple output elements per thread
__global__ void q4_0_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_blocks = K / 32;

    // Each thread processes 2 output elements (TN=2)
    const int TN = 2;
    const int m_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_base = blockIdx.x * blockDim.x * TN + threadIdx.x * TN;

    if (m_idx >= M) return;

    const float* act_row = activation + m_idx * K;

    float accum[TN] = {0.0f, 0.0f};

    for (int block = 0; block < K_blocks; block++) {
        const float* act_ptr = act_row + block * 32;

        // Load 32 activations once
        float a[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a[i] = act_ptr[i];
        }

        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int n_idx = n_base + tn;
            if (n_idx >= N) continue;

            const uint8_t* w_block = weight_q + n_idx * K_blocks * 18 + block * 18;

            // Load scale
            half2 scale_data = *reinterpret_cast<const half2*>(w_block);
            float d_w = __half2float(scale_data.x);

            const uint8_t* qs_ptr = w_block + 2;

            // Unroll the dot product for better performance
            float block_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t byte = qs_ptr[i];
                int q_low = byte & 0x0F;
                int q_high = (byte >> 4) & 0x0F;

                float w_low = d_w * (float(q_low) - 8.0f);
                float w_high = d_w * (float(q_high) - 8.0f);

                block_sum += w_low * a[i] + w_high * a[i + 16];
            }
            accum[tn] += block_sum;
        }
    }

    // Write results
    #pragma unroll
    for (int tn = 0; tn < TN; tn++) {
        int n_idx = n_base + tn;
        if (n_idx < N) {
            output[m_idx * N + n_idx] = accum[tn];
        }
    }
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

    // Optimize block sizes based on M and N
    const int TN = 2;
    int threads_x = 16;
    int threads_y = 16;

    dim3 block(threads_x, threads_y);
    dim3 grid((N + threads_x * TN - 1) / (threads_x * TN), (M + threads_y - 1) / threads_y);
    q4_0_fp32_gemm_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
