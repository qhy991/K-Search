// W4A32C8 Q4_0 GEMM Kernel for DeepSeek-V2 Attention Output
// Dimensions: M (batch), N=5120 (output), K=5120 (input)
// Weight: Q4_0 quantized [N, K/32], Activation: FP32 [M, K]
// Output: FP32 [M, N]
//
// Formula: result = sum(scale_w * (q_w - 8) * activation)
// where Q4_0 uses offset-8 encoding
//
// v4: Compute-optimized with ILP, vectorized loads, and loop unrolling

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Q4_0 block format
#define BLOCK_K 32  // quantization block size
#define Q4_0_BLOCK_SIZE 18  // bytes per q4_0 block

// Helper: Convert FP16 to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Helper: Load FP16 from bytes
__device__ __forceinline__ half load_half(const uint8_t* ptr) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(ptr);
    return __ushort_as_half(val);
}

// Optimized kernel: Process 4 outputs per thread block for better ILP
// Each thread computes partial sums for multiple K chunks
__global__ void q4_0_gemm_optimized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    // Each thread processes one (m, n) output element
    const int m = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    // Get pointers
    const float* act_row = activation + m * K;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    // Process K in chunks of 4 blocks for ILP (128 values per iteration)
    const int chunks = num_blocks_k / 4;
    const int remainder = num_blocks_k % 4;

    // Main loop: process 4 blocks at a time
    for (int c = 0; c < chunks; c++) {
        int kb = c * 4;

        // Load 4 scales at once
        half d_h0 = load_half(w_row + (kb + 0) * Q4_0_BLOCK_SIZE);
        half d_h1 = load_half(w_row + (kb + 1) * Q4_0_BLOCK_SIZE);
        half d_h2 = load_half(w_row + (kb + 2) * Q4_0_BLOCK_SIZE);
        half d_h3 = load_half(w_row + (kb + 3) * Q4_0_BLOCK_SIZE);

        float d_w0 = half_to_float(d_h0);
        float d_w1 = half_to_float(d_h1);
        float d_w2 = half_to_float(d_h2);
        float d_w3 = half_to_float(d_h3);

        // Get packed data pointers
        const uint8_t* w_packed0 = w_row + (kb + 0) * Q4_0_BLOCK_SIZE + 2;
        const uint8_t* w_packed1 = w_row + (kb + 1) * Q4_0_BLOCK_SIZE + 2;
        const uint8_t* w_packed2 = w_row + (kb + 2) * Q4_0_BLOCK_SIZE + 2;
        const uint8_t* w_packed3 = w_row + (kb + 3) * Q4_0_BLOCK_SIZE + 2;

        // Process 4 blocks with full unrolling for ILP
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            // Block 0
            uint8_t b0 = w_packed0[i];
            int w0_0 = b0 & 0x0F;
            int w0_1 = (b0 >> 4) & 0x0F;
            int k0_0 = (kb + 0) * BLOCK_K + i;
            int k0_1 = k0_0 + 16;
            sum += act_row[k0_0] * d_w0 * (w0_0 - 8);
            sum += act_row[k0_1] * d_w0 * (w0_1 - 8);

            // Block 1
            uint8_t b1 = w_packed1[i];
            int w1_0 = b1 & 0x0F;
            int w1_1 = (b1 >> 4) & 0x0F;
            int k1_0 = (kb + 1) * BLOCK_K + i;
            int k1_1 = k1_0 + 16;
            sum += act_row[k1_0] * d_w1 * (w1_0 - 8);
            sum += act_row[k1_1] * d_w1 * (w1_1 - 8);

            // Block 2
            uint8_t b2 = w_packed2[i];
            int w2_0 = b2 & 0x0F;
            int w2_1 = (b2 >> 4) & 0x0F;
            int k2_0 = (kb + 2) * BLOCK_K + i;
            int k2_1 = k2_0 + 16;
            sum += act_row[k2_0] * d_w2 * (w2_0 - 8);
            sum += act_row[k2_1] * d_w2 * (w2_1 - 8);

            // Block 3
            uint8_t b3 = w_packed3[i];
            int w3_0 = b3 & 0x0F;
            int w3_1 = (b3 >> 4) & 0x0F;
            int k3_0 = (kb + 3) * BLOCK_K + i;
            int k3_1 = k3_0 + 16;
            sum += act_row[k3_0] * d_w3 * (w3_0 - 8);
            sum += act_row[k3_1] * d_w3 * (w3_1 - 8);
        }
    }

    // Handle remainder blocks
    for (int kb = chunks * 4; kb < num_blocks_k; kb++) {
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

// Warp-based kernel with shared memory for activation caching
__global__ void q4_0_gemm_warp_shared(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    // Each warp processes one (m, n) output element
    const int m = blockIdx.x;
    const int n = blockIdx.y;

    if (m >= M || n >= N) return;

    const int lane_id = threadIdx.x % 32;

    // Shared memory for activation tile
    __shared__ float act_smem[5120];  // K=5120

    // Cooperatively load activation row into shared memory
    const float* act_row = activation + m * K;
    for (int i = threadIdx.x; i < K; i += 32) {
        act_smem[i] = act_row[i];
    }
    __syncthreads();

    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    // Process all K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load weight scale
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);

        // Get packed weights
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        // Each lane processes one element
        int w_q;
        int k_idx;
        if (lane_id < 16) {
            w_q = w_packed[lane_id] & 0x0F;
            k_idx = kb * BLOCK_K + lane_id;
        } else {
            w_q = (w_packed[lane_id - 16] >> 4) & 0x0F;
            k_idx = kb * BLOCK_K + lane_id;
        }

        float w_val = d_w * (w_q - 8);
        sum += act_smem[k_idx] * w_val;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write result
    if (lane_id == 0) {
        output[m * N + n] = sum;
    }
}

// Single token optimized kernel with vectorized loads
__global__ void q4_0_gemm_single_optimized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    // Each thread processes one output element
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // M=1, single activation row
    const float* act_row = activation;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    // Process K in chunks of 4 blocks for ILP
    const int chunks = num_blocks_k / 4;
    const int remainder = num_blocks_k % 4;

    for (int c = 0; c < chunks; c++) {
        int kb = c * 4;

        // Load 4 scales
        half d_h0 = load_half(w_row + (kb + 0) * Q4_0_BLOCK_SIZE);
        half d_h1 = load_half(w_row + (kb + 1) * Q4_0_BLOCK_SIZE);
        half d_h2 = load_half(w_row + (kb + 2) * Q4_0_BLOCK_SIZE);
        half d_h3 = load_half(w_row + (kb + 3) * Q4_0_BLOCK_SIZE);

        float d_w0 = half_to_float(d_h0);
        float d_w1 = half_to_float(d_h1);
        float d_w2 = half_to_float(d_h2);
        float d_w3 = half_to_float(d_h3);

        const uint8_t* w_packed0 = w_row + (kb + 0) * Q4_0_BLOCK_SIZE + 2;
        const uint8_t* w_packed1 = w_row + (kb + 1) * Q4_0_BLOCK_SIZE + 2;
        const uint8_t* w_packed2 = w_row + (kb + 2) * Q4_0_BLOCK_SIZE + 2;
        const uint8_t* w_packed3 = w_row + (kb + 3) * Q4_0_BLOCK_SIZE + 2;

        // Process 4 blocks unrolled
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t b0 = w_packed0[i], b1 = w_packed1[i];
            uint8_t b2 = w_packed2[i], b3 = w_packed3[i];

            int w0_0 = b0 & 0x0F, w0_1 = (b0 >> 4) & 0x0F;
            int w1_0 = b1 & 0x0F, w1_1 = (b1 >> 4) & 0x0F;
            int w2_0 = b2 & 0x0F, w2_1 = (b2 >> 4) & 0x0F;
            int w3_0 = b3 & 0x0F, w3_1 = (b3 >> 4) & 0x0F;

            int k_base = (kb + 0) * BLOCK_K + i;
            sum += act_row[k_base] * d_w0 * (w0_0 - 8) + act_row[k_base + 16] * d_w0 * (w0_1 - 8);

            k_base = (kb + 1) * BLOCK_K + i;
            sum += act_row[k_base] * d_w1 * (w1_0 - 8) + act_row[k_base + 16] * d_w1 * (w1_1 - 8);

            k_base = (kb + 2) * BLOCK_K + i;
            sum += act_row[k_base] * d_w2 * (w2_0 - 8) + act_row[k_base + 16] * d_w2 * (w2_1 - 8);

            k_base = (kb + 3) * BLOCK_K + i;
            sum += act_row[k_base] * d_w3 * (w3_0 - 8) + act_row[k_base + 16] * d_w3 * (w3_1 - 8);
        }
    }

    // Handle remainder
    for (int kb = chunks * 4; kb < num_blocks_k; kb++) {
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t byte = w_packed[i];
            int w0 = byte & 0x0F, w1 = (byte >> 4) & 0x0F;
            int k0 = kb * BLOCK_K + i;
            sum += act_row[k0] * d_w * (w0 - 8) + act_row[k0 + 16] * d_w * (w1 - 8);
        }
    }

    output[n] = sum;
}

// Dispatch wrapper
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
        // Single token: optimized single-token kernel
        int threads = min(256, (int)N);
        int blocks = (N + threads - 1) / threads;
        q4_0_gemm_single_optimized<<<blocks, threads>>>(
            weight_ptr, act_ptr, output_ptr, (int)N, (int)K
        );
    } else if (M >= 8) {
        // Larger batch: optimized thread-per-output kernel
        int threads_per_block = min(256, (int)N);
        dim3 block(threads_per_block);
        dim3 grid(M, (N + threads_per_block - 1) / threads_per_block);
        q4_0_gemm_optimized<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, (int)M, (int)N, (int)K
        );
    } else {
        // Medium batch: warp-based kernel with shared memory
        dim3 block(32);
        dim3 grid(M, N);
        q4_0_gemm_warp_shared<<<grid, block>>>(
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
    m.def("forward", &forward, "Q4_0 GEMM Forward - DeepSeek-V2 Attention Output v4");
}
