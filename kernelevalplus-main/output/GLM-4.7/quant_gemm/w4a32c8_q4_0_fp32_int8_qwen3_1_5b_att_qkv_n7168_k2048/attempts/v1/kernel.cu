#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 block structure
struct block_q4_0 {
    uint16_t d;      // FP16 scale
    uint8_t qs[16];  // 32 packed quaternions
};

// Convert FP16 to FP32 using union for safety
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Optimized vectorized dot product for Q4_0 block x FP32 activation
__device__ __forceinline__ float q4_0_dot_fp32(
    const block_q4_0& wb,
    const float* __restrict__ act_row,
    int k_base
) {
    float d_w = fp16_to_fp32(wb.d);
    float sum = 0.0f;

    // Unroll and vectorize
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t packed = wb.qs[i];
        int q0 = packed & 0x0F;
        int q1 = (packed >> 4) & 0x0F;
        int q2 = wb.qs[i + 8] & 0x0F;
        int q3 = (wb.qs[i + 8] >> 4) & 0x0F;

        float a0 = act_row[k_base + i];
        float a1 = act_row[k_base + i + 8];
        float a2 = act_row[k_base + i + 16];
        float a3 = act_row[k_base + i + 24];

        sum += d_w * (q0 - 8) * a0;
        sum += d_w * (q1 - 8) * a1;
        sum += d_w * (q2 - 8) * a2;
        sum += d_w * (q3 - 8) * a3;
    }

    return sum;
}

// Small batch kernel (M <= 8): One thread per output element
// Optimize for low-latency single-token inference
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

    // Each thread processes one output element
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0& wb = weight[n * num_blocks_k + kb];
        sum += q4_0_dot_fp32(wb, act_row, kb * 32);
    }

    output[m * N + n] = sum;
}

// Large batch kernel (M > 8): 8x8 thread blocks with shared memory
__global__ void __launch_bounds__(64) q4_0_fp32_gemm_large_batch(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int tx = threadIdx.x;  // 0-7
    const int ty = threadIdx.y;  // 0-7

    const int m = blockIdx.y * 8 + tx;
    const int n = blockIdx.x * 8 + ty;

    if (m >= M || n >= N) return;

    const float* act_row = activation + m * K;
    float sum = 0.0f;

    // Each thread computes one output element
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0& wb = weight[n * num_blocks_k + kb];
        sum += q4_0_dot_fp32(wb, act_row, kb * 32);
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
    } else {
        // Large batch: 8x8 thread blocks
        dim3 grid((N + 7) / 8, (M + 7) / 8);
        dim3 block(8, 8);

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
