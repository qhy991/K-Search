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
// llama.cpp packing: qs[0..15] = low nibbles (positions 0-15)
//                    qs[16..31] = high nibbles (positions 16-31)
__device__ __forceinline__ void unpack_q4_0_4(
    const uint8_t* __restrict__ qs,
    int i,
    int* q0, int* q1, int* q2, int* q3
) {
    uint8_t packed0 = qs[i];
    uint8_t packed1 = qs[i + 8];
    *q0 = packed0 & 0x0F;        // position i
    *q1 = (packed0 >> 4) & 0x0F;  // position i + 16
    *q2 = packed1 & 0x0F;        // position i + 8
    *q3 = (packed1 >> 4) & 0x0F;  // position i + 24
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

// Small batch kernel (M <= 8): One thread per output element
__global__ void __launch_bounds__(256) q4_0_fp32_gemm_small_batch(
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

// Medium batch kernel - 8x4 thread blocks
__global__ void __launch_bounds__(32) q4_0_fp32_gemm_medium_batch(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int tx = threadIdx.x;  // 0-31
    const int ty = threadIdx.y;  // 0-3
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

// Large batch kernel - 1D blocks
__global__ void __launch_bounds__(256) q4_0_fp32_gemm_large_batch(
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

    // Choose kernel based on batch size
    if (M <= 8) {
        // Small batch: one thread per output element
        const int threads_per_block = 256;
        dim3 grid((N + threads_per_block - 1) / threads_per_block, M);
        dim3 block(threads_per_block);

        q4_0_fp32_gemm_small_batch<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 64) {
        // Medium batch: 32x4 blocks
        const int threads_x = 32;
        const int threads_y = 4;
        dim3 grid((N + threads_x - 1) / threads_x, (M + threads_y - 1) / threads_y);
        dim3 block(threads_x, threads_y);

        q4_0_fp32_gemm_medium_batch<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: 256 threads per block
        const int threads_per_block = 256;
        dim3 grid((N + threads_per_block - 1) / threads_per_block, M);
        dim3 block(threads_per_block);

        q4_0_fp32_gemm_large_batch<<<grid, block>>>(
            reinterpret_cast<const block_q4_0*>(weight.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM with batch-dependent dispatch");
}
