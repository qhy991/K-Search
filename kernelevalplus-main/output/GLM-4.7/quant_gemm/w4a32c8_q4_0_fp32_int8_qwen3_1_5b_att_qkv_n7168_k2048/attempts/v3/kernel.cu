#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 block structure
struct block_q4_0 {
    uint16_t d;      // FP16 scale
    uint8_t qs[16];  // 32 packed quaternions
};

// Convert FP16 to FP32 using intrinsic
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    return __half2float(*reinterpret_cast<__half*>(&h));
}

// Unpack 4 Q4_0 values from packed bytes
__device__ __forceinline__ void unpack_q4_0_4(
    const uint8_t* __restrict__ qs,
    int i,
    int* q0, int* q1, int* q2, int* q3
) {
    uint8_t packed0 = qs[i];
    uint8_t packed1 = qs[i + 8];
    *q0 = packed0 & 0x0F;
    *q1 = (packed0 >> 4) & 0x0F;
    *q2 = packed1 & 0x0F;
    *q3 = (packed1 >> 4) & 0x0F;
}

// Compute dot product for one Q4_0 block with FP32 activation
__device__ __forceinline__ float q4_0_block_dot_fp32(
    const block_q4_0& wb,
    const float* __restrict__ act,
    int k_base
) {
    float d_w = fp16_to_fp32(wb.d);
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int q0, q1, q2, q3;
        unpack_q4_0_4(wb.qs, i, &q0, &q1, &q2, &q3);

        sum += d_w * (q0 - 8) * act[k_base + i];
        sum += d_w * (q1 - 8) * act[k_base + i + 16];
        sum += d_w * (q2 - 8) * act[k_base + i + 8];
        sum += d_w * (q3 - 8) * act[k_base + i + 24];
    }

    return sum;
}

// Small batch kernel - simple and efficient
__global__ void __launch_bounds__(1024) q4_0_fp32_gemm_small(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    const float* act_row = activation + m * K;
    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0& wb = weight[n * num_blocks_k + kb];
        sum += q4_0_block_dot_fp32(wb, act_row, kb * 32);
    }

    output[m * N + n] = sum;
}

// Medium batch kernel - 2D blocks
__global__ void __launch_bounds__(256) q4_0_fp32_gemm_medium(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * 32;
    const int by = blockIdx.y * 4;

    const int n = bx + tx;
    const int m = by + ty;

    if (m >= M || n >= N) return;

    const float* act_row = activation + m * K;
    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0& wb = weight[n * num_blocks_k + kb];
        sum += q4_0_block_dot_fp32(wb, act_row, kb * 32);
    }

    output[m * N + n] = sum;
}

// Large batch kernel
__global__ void __launch_bounds__(256) q4_0_fp32_gemm_large(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int tx = threadIdx.x;
    const int bx = blockIdx.x * 256;
    const int by = blockIdx.y;

    const int n = bx + tx;
    const int m = by;

    if (m >= M || n >= N) return;

    const float* act_row = activation + m * K;
    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0& wb = weight[n * num_blocks_k + kb];
        sum += q4_0_block_dot_fp32(wb, act_row, kb * 32);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    if (M <= 8) {
        // Small batch: use large thread blocks for better GPU utilization
        const int threads = 1024;
        dim3 grid((N + threads - 1) / threads, M);
        dim3 block(threads);

        q4_0_fp32_gemm_small<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 64) {
        dim3 grid((N + 31) / 32, (M + 3) / 4);
        dim3 block(32, 4);

        q4_0_fp32_gemm_medium<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 grid((N + 255) / 256, M);
        dim3 block(256);

        q4_0_fp32_gemm_large<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
