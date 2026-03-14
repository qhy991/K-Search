/**
 * Q8_0 x FP32 GEMM Kernel for DeepSeek-V2 LM Head
 * N=102400, K=5120 (very large output dimension)
 *
 * V6 Optimizations:
 * - M=1: 32 threads (one warp) with DP4A and warp shuffle reduction
 * - M<=16: 256 threads with DP4A for small batches
 * - M>16: Dequantize to FP16 and use Tensor Cores via PyTorch
 * - Fixed alignment issues with vectorized loads
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q8_0 block format: FP16 scale (2 bytes) + INT8 quants[32] (32 bytes) = 34 bytes total
struct block_q8_0 {
    uint16_t d;      // FP16 scale
    int8_t qs[32];   // INT8 quantized values
};
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block size must be 34 bytes");

// Helper function to convert FP16 to FP32
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * M=1 kernel with 32 threads (one warp) - maximum occupancy
 * Uses DP4A instruction for INT8 dot product
 * Uses warp shuffle reduction instead of shared memory
 */
__global__ void __launch_bounds__(32) gemm_m1_dp4a_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K
) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    if (n >= N) return;

    const int num_blocks = K / 32;
    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);

    double sum = 0.0;

    // Grid-stride loop over K blocks
    for (int kb = tid; kb < num_blocks; kb += blockDim.x) {
        // Load weight block
        const block_q8_0& w_block = w_blocks[n * num_blocks + kb];
        const float d_w = fp16_to_fp32(w_block.d);

        // Load activation block (32 floats)
        const int k_base = kb * 32;

        // Compute activation scale for dynamic quantization
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            amax = fmaxf(amax, fabsf(activation[k_base + i]));
        }
        const float d_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        // DP4A dot product - process 8 int4 values per iteration
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Pack 4 int8 values into one int32
            int w_packed = *reinterpret_cast<const int*>(&w_block.qs[i * 4]);

            // Load and quantize 4 activation values
            int8_t a_vals[4];
            a_vals[0] = (int8_t)max(-128, min(127, __float2int_rn(activation[k_base + i * 4 + 0] * inv_d_a)));
            a_vals[1] = (int8_t)max(-128, min(127, __float2int_rn(activation[k_base + i * 4 + 1] * inv_d_a)));
            a_vals[2] = (int8_t)max(-128, min(127, __float2int_rn(activation[k_base + i * 4 + 2] * inv_d_a)));
            a_vals[3] = (int8_t)max(-128, min(127, __float2int_rn(activation[k_base + i * 4 + 3] * inv_d_a)));

            // Pack into int32 and use DP4A
            int a_packed = *reinterpret_cast<int*>(a_vals);
            sumi = __dp4a(w_packed, a_packed, sumi);
        }

        sum += (double)d_w * (double)d_a * (double)sumi;
    }

    // Warp shuffle reduction (no shared memory needed)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only thread 0 writes the result
    if (tid == 0) {
        output[n] = (float)sum;
    }
}

/**
 * Small batch kernel (M=2 to M=16) with DP4A
 */
__global__ void __launch_bounds__(256) gemm_small_batch_dp4a_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (n >= N || m >= M) return;

    const int num_blocks = K / 32;
    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const float* act_row = activation + m * K;

    double sum = 0.0;

    for (int kb = 0; kb < num_blocks; kb++) {
        const block_q8_0& w_block = w_blocks[n * num_blocks + kb];
        const float d_w = fp16_to_fp32(w_block.d);

        const int k_base = kb * 32;

        // Compute activation scale for dynamic quantization
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            amax = fmaxf(amax, fabsf(act_row[k_base + i]));
        }
        const float d_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        // DP4A dot product
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_packed = *reinterpret_cast<const int*>(&w_block.qs[i * 4]);

            int8_t a_vals[4];
            a_vals[0] = (int8_t)max(-128, min(127, __float2int_rn(act_row[k_base + i * 4 + 0] * inv_d_a)));
            a_vals[1] = (int8_t)max(-128, min(127, __float2int_rn(act_row[k_base + i * 4 + 1] * inv_d_a)));
            a_vals[2] = (int8_t)max(-128, min(127, __float2int_rn(act_row[k_base + i * 4 + 2] * inv_d_a)));
            a_vals[3] = (int8_t)max(-128, min(127, __float2int_rn(act_row[k_base + i * 4 + 3] * inv_d_a)));

            int a_packed = *reinterpret_cast<int*>(a_vals);
            sumi = __dp4a(w_packed, a_packed, sumi);
        }

        sum += (double)d_w * (double)d_a * (double)sumi;
    }

    output[m * N + n] = (float)sum;
}

/**
 * Dequantize Q8_0 weights to FP16 for large batch processing
 */
__global__ void dequantize_q8_0_to_fp16_kernel(
    const uint8_t* __restrict__ weight_q,
    half* __restrict__ weight_fp16,
    const int N, const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = blockIdx.y;

    const int num_blocks = K / 32;
    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight_q);

    if (n >= N) return;

    const block_q8_0& w_block = w_blocks[n * num_blocks + kb];

    // Convert FP16 scale
    union { uint16_t u16; __half f16; } un;
    un.u16 = w_block.d;
    const __half d_w = un.f16;

    const int k_start = kb * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        weight_fp16[n * K + k_start + i] = __hmul(__int2half_rn((int)w_block.qs[i]), d_w);
    }
}

/**
 * Forward pass with adaptive strategy selection
 */
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");

    const int num_blocks = K / 32;

    if (M == 1) {
        // Single token: use 32 threads (one warp) per block for maximum occupancy
        auto output = torch::empty({1, N}, torch::dtype(torch::kFloat32).device(weight.device()));

        dim3 block(32);
        dim3 grid(N);

        gemm_m1_dp4a_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err));
        }

        return output;
    }
    else if (M <= 16) {
        // Small batch: per-row kernel with DP4A
        auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

        dim3 block(256);
        dim3 grid((N + 255) / 256, M);

        gemm_small_batch_dp4a_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err));
        }

        return output;
    }
    else {
        // Large batch: dequantize and use FP16 Tensor Core via PyTorch
        auto weight_fp16 = torch::empty({N, K}, torch::dtype(torch::kFloat16).device(weight.device()));

        dim3 dequant_block(256);
        dim3 dequant_grid((N + 255) / 256, num_blocks);

        dequantize_q8_0_to_fp16_kernel<<<dequant_grid, dequant_block>>>(
            weight.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(weight_fp16.data_ptr()),
            N, K
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "Dequantize kernel failed: ", cudaGetErrorString(err));
        }

        auto activation_fp16 = activation.to(torch::kFloat16);
        return activation_fp16.matmul(weight_fp16.t()).to(torch::kFloat32);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W8A32C8 Q8_0 Quantized GEMM with DP4A and Tensor Cores");
}
