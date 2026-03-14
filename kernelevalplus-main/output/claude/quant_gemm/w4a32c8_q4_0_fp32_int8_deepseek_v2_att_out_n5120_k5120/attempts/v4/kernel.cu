#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 format: 18 bytes per block
// 2 bytes: FP16 scale
// 16 bytes: 32 packed 4-bit values

// Optimized kernel with better parallelism for small batch sizes
__global__ void q4_0_fp32_gemm_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_blocks = K / 32;
    const int warp_size = 32;

    // Each warp processes multiple output elements
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warp_size;
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % warp_size;

    const int m_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_base = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    if (m_idx >= M) return;

    const float* act_row = activation + m_idx * K;

    // Accumulate for 4 output elements per thread
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block = 0; block < K_blocks; block++) {
        // Load activation block into shared memory
        __shared__ float s_act[32];
        const float* act_ptr = act_row + block * 32;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        if (tid < 32) s_act[tid] = act_ptr[tid];
        __syncthreads();

        #pragma unroll
        for (int tn = 0; tn < 4; tn++) {
            int n_idx = n_base + tn;
            if (n_idx >= N) continue;

            const uint8_t* w_block = weight_q + n_idx * K_blocks * 18 + block * 18;

            // Load scale
            half2 scale_data = *reinterpret_cast<const half2*>(w_block);
            float d_w = __half2float(scale_data.x);

            const uint8_t* qs_ptr = w_block + 2;

            // Compute dot product with shared activation
            float block_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = qs_ptr[i];
                int q_low = byte & 0x0F;
                int q_high = (byte >> 4) & 0x0F;

                float w_low = d_w * (float(q_low) - 8.0f);
                float w_high = d_w * (float(q_high) - 8.0f);

                block_sum += w_low * s_act[i];
                block_sum += w_high * s_act[i + 16];
            }
            accum[tn] += block_sum;
        }
        __syncthreads();
    }

    // Write results
    for (int tn = 0; tn < 4; tn++) {
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

    // Optimize for different batch sizes
    if (M == 1) {
        // Special case for M=1: maximize parallelism across N
        int threads = 256;
        int outputs_per_thread = 4;
        dim3 block(threads);
        dim3 grid((N + threads * outputs_per_thread - 1) / (threads * outputs_per_thread));
        q4_0_fp32_gemm_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);
    } else {
        // General case
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        q4_0_fp32_gemm_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
