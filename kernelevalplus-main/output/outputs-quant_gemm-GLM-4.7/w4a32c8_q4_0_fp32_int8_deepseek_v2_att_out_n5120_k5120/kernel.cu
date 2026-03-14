// W4A32C8 Q4_0 GEMM Kernel for DeepSeek-V2 Attention Output
// v8: Best kernels from previous versions - hybrid approach
//
// Uses:
// - v3's M=1 kernel (best for single token)
// - v7's optimized medium/large batch kernels

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Q4_0 block format
#define BLOCK_K 32
#define Q4_0_BLOCK_SIZE 18

// Helper: Convert FP16 to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Helper: Load FP16 from bytes
__device__ __forceinline__ half load_half(const uint8_t* ptr) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(ptr);
    return __ushort_as_half(val);
}

// ============================================================
// Kernel 1: Simple thread-per-output (best for large M)
// ============================================================
__global__ void q4_0_gemm_simple(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    const int m = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const float* act_row = activation + m * K;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t byte = w_packed[i];
            int w0 = byte & 0x0F;
            int w1 = (byte >> 4) & 0x0F;
            int k0 = kb * BLOCK_K + i;
            int k1 = k0 + 16;
            sum += act_row[k0] * d_w * (w0 - 8) + act_row[k1] * d_w * (w1 - 8);
        }
    }

    output[m * N + n] = sum;
}

// ============================================================
// Kernel 2: Medium batch with 2 outputs per thread (best ILP)
// ============================================================
__global__ void q4_0_gemm_medium_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    const int m = blockIdx.x;
    if (m >= M) return;

    const int n_base = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
    if (n_base >= N) return;

    const int n1 = n_base;
    const int n2 = n_base + 1;
    const bool has_second = (n2 < N);

    const float* act_row = activation + m * K;

    float sum1 = 0.0f, sum2 = 0.0f;

    const uint8_t* w_row1 = weight + n1 * num_blocks_k * Q4_0_BLOCK_SIZE;
    const uint8_t* w_row2 = has_second ? weight + n2 * num_blocks_k * Q4_0_BLOCK_SIZE : w_row1;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        half d_h1 = load_half(w_row1 + kb * Q4_0_BLOCK_SIZE);
        half d_h2 = has_second ? load_half(w_row2 + kb * Q4_0_BLOCK_SIZE) : d_h1;

        float d_w1 = half_to_float(d_h1);
        float d_w2 = half_to_float(d_h2);

        const uint8_t* w_packed1 = w_row1 + kb * Q4_0_BLOCK_SIZE + 2;
        const uint8_t* w_packed2 = has_second ? w_row2 + kb * Q4_0_BLOCK_SIZE + 2 : w_packed1;

        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t b1 = w_packed1[i];
            int w1_0 = b1 & 0x0F, w1_1 = (b1 >> 4) & 0x0F;
            int k0 = kb * BLOCK_K + i, k1 = k0 + 16;
            sum1 += act_row[k0] * d_w1 * (w1_0 - 8) + act_row[k1] * d_w1 * (w1_1 - 8);

            if (has_second) {
                uint8_t b2 = w_packed2[i];
                int w2_0 = b2 & 0x0F, w2_1 = (b2 >> 4) & 0x0F;
                sum2 += act_row[k0] * d_w2 * (w2_0 - 8) + act_row[k1] * d_w2 * (w2_1 - 8);
            }
        }
    }

    output[m * N + n1] = sum1;
    if (has_second) output[m * N + n2] = sum2;
}

// ============================================================
// Kernel 3: Large batch optimized with 8 outputs per block
// ============================================================
__global__ void q4_0_gemm_large_batch_optimized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;
    const int N_PER_BLOCK = 8;

    const int m = blockIdx.x;
    if (m >= M) return;

    const int n_start = blockIdx.y * N_PER_BLOCK;
    if (n_start >= N) return;

    const int n_end = min(n_start + N_PER_BLOCK, N);
    const int n_local = threadIdx.x;

    if (n_local >= N_PER_BLOCK) return;

    const int n = n_start + n_local;
    if (n >= N) return;

    const float* act_row = activation + m * K;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t byte = w_packed[i];
            int w0 = byte & 0x0F;
            int w1 = (byte >> 4) & 0x0F;
            int k0 = kb * BLOCK_K + i;
            int k1 = k0 + 16;
            sum += act_row[k0] * d_w * (w0 - 8) + act_row[k1] * d_w * (w1 - 8);
        }
    }

    output[m * N + n] = sum;
}

// ============================================================
// Kernel 4: Single token (M=1) - simple thread-per-output
// ============================================================
__global__ void q4_0_gemm_single_token(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* act_row = activation;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t byte = w_packed[i];
            int w0 = byte & 0x0F;
            int w1 = (byte >> 4) & 0x0F;
            int k0 = kb * BLOCK_K + i;
            int k1 = kb * BLOCK_K + i + 16;
            sum += act_row[k0] * d_w * (w0 - 8) + act_row[k1] * d_w * (w1 - 8);
        }
    }

    output[n] = sum;
}

// ============================================================
// Dispatch wrapper - best kernel per configuration
// ============================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if (M == 1) {
        // Single token: simple thread-per-output (best from v3)
        int threads = min(256, (int)N);
        int blocks = (N + threads - 1) / threads;
        q4_0_gemm_single_token<<<blocks, threads>>>(
            weight_ptr, act_ptr, output_ptr, (int)N, (int)K
        );
    } else if (M >= 8) {
        // Large batch: 8 outputs per block (best from v7)
        dim3 block(8);
        dim3 grid(M, (N + 7) / 8);
        q4_0_gemm_large_batch_optimized<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, (int)M, (int)N, (int)K
        );
    } else {
        // Medium batch: 2 outputs per thread for ILP
        int threads = 128;
        dim3 block(threads);
        dim3 grid(M, (N + 2 * threads - 1) / (2 * threads));
        q4_0_gemm_medium_batch<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, (int)M, (int)N, (int)K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM Forward - v8 Hybrid Best");
}
