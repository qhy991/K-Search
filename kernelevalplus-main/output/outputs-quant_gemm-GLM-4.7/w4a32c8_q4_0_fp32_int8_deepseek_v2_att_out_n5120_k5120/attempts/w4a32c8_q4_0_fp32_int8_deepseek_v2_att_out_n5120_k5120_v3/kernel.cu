// W4A32C8 Q4_0 GEMM Kernel for DeepSeek-V2 Attention Output
// Dimensions: M (batch), N=5120 (output), K=5120 (input)
// Weight: Q4_0 quantized [N, K/32], Activation: FP32 [M, K]
// Output: FP32 [M, N]
//
// Formula: result = sum(scale_w * (q_w - 8) * activation)
// where Q4_0 uses offset-8 encoding
//
// Compute-optimized implementation focusing on correctness and efficiency

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Q4_0 block format
#define BLOCK_K 32  // quantization block size
#define Q4_0_BLOCK_SIZE 18  // bytes per q4_0 block (2-byte FP16 scale + 16 bytes packed data)

// Helper: Convert FP16 to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Helper: Load FP16 from bytes
__device__ __forceinline__ half load_half(const uint8_t* ptr) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(ptr);
    return __ushort_as_half(val);
}

// Kernel 1: Simple thread-per-output kernel (works for all M sizes)
// Each thread computes one output element
__global__ void q4_0_gemm_simple(
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

    // Get pointers for this row
    const float* act_row = activation + m * K;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    // Process all K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load weight scale (FP16)
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);

        // Get packed weights (16 bytes containing 32 x 4-bit values)
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        // Process 16 bytes = 32 values
        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t byte = w_packed[i];
            int w0 = byte & 0x0F;           // low nibble (position i)
            int w1 = (byte >> 4) & 0x0F;    // high nibble (position i + 16)

            int k0 = kb * BLOCK_K + i;      // position i
            int k1 = kb * BLOCK_K + i + 16; // position i + 16

            // Dequantize: Q4_0 encoding is offset-8
            // q = round(val / scale + 8), so val = scale * (q - 8)
            float w_val0 = d_w * (w0 - 8);
            float w_val1 = d_w * (w1 - 8);

            sum += act_row[k0] * w_val0 + act_row[k1] * w_val1;
        }
    }

    output[m * N + n] = sum;
}

// Kernel 2: Warp-based kernel for better utilization (one warp per output)
__global__ void q4_0_gemm_warp(
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

    // Get pointers for this row
    const float* act_row = activation + m * K;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    // Process all K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load weight scale
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);

        // Get packed weights
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        // Each lane processes one element from the 32-value block
        // Lanes 0-15: positions 0-15 (low nibbles)
        // Lanes 16-31: positions 16-31 (high nibbles)
        int w_q;
        int k_offset;
        if (lane_id < 16) {
            w_q = w_packed[lane_id] & 0x0F;
            k_offset = lane_id;
        } else {
            w_q = (w_packed[lane_id - 16] >> 4) & 0x0F;
            k_offset = lane_id;
        }

        int k_idx = kb * BLOCK_K + k_offset;
        float w_val = d_w * (w_q - 8);
        sum += act_row[k_idx] * w_val;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write result (lane 0 only)
    if (lane_id == 0) {
        output[m * N + n] = sum;
    }
}

// Kernel 3: Block-tiled kernel for better weight reuse
// Each block processes multiple N columns for one M row
__global__ void q4_0_gemm_tiled(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;
    const int N_PER_BLOCK = 32;  // Process 32 N columns per block

    // Block configuration
    const int m = blockIdx.x;
    if (m >= M) return;

    const int n_start = blockIdx.y * N_PER_BLOCK;
    if (n_start >= N) return;

    const int n_end = min(n_start + N_PER_BLOCK, N);
    const int n_local = threadIdx.x;  // 0 to 31

    const int n = n_start + n_local;
    const bool valid_n = (n < n_end);

    // Get pointers
    const float* act_row = activation + m * K;

    float sum = 0.0f;

    // Process all K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        if (valid_n) {
            // Get weight row for this N column
            const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

            // Load weight scale
            half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
            float d_w = half_to_float(d_h);

            // Get packed weights
            const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

            // Process 16 bytes = 32 values
            #pragma unroll 2
            for (int i = 0; i < 16; i++) {
                uint8_t byte = w_packed[i];
                int w0 = byte & 0x0F;
                int w1 = (byte >> 4) & 0x0F;

                int k0 = kb * BLOCK_K + i;
                int k1 = kb * BLOCK_K + i + 16;

                sum += act_row[k0] * d_w * (w0 - 8) + act_row[k1] * d_w * (w1 - 8);
            }
        }
    }

    // Write result
    if (valid_n) {
        output[m * N + n] = sum;
    }
}

// Kernel 4: Optimized single token kernel using vectorized loads
__global__ void q4_0_gemm_single_token(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    // Each thread processes one output element (M=1)
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // M=1, so activation is just a 1D array of K elements
    const float* act_row = activation;
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    // Process all K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load weight scale
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);

        // Get packed weights
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

        // Process 32 values in this block
        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t byte = w_packed[i];
            int w0 = byte & 0x0F;
            int w1 = (byte >> 4) & 0x0F;

            int k0 = kb * BLOCK_K + i;
            int k1 = kb * BLOCK_K + i + 16;

            // Dequantize: val = scale * (q - 8)
            float w_val0 = d_w * (w0 - 8);
            float w_val1 = d_w * (w1 - 8);

            sum += act_row[k0] * w_val0 + act_row[k1] * w_val1;
        }
    }

    output[n] = sum;
}

// Dispatch wrapper - selects optimal kernel based on configuration
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

    // Select kernel based on M (batch size)
    if (M == 1) {
        // Single token: use dedicated single-token kernel
        int threads = min(256, (int)N);
        int blocks = (N + threads - 1) / threads;
        q4_0_gemm_single_token<<<blocks, threads>>>(
            weight_ptr, act_ptr, output_ptr, (int)N, (int)K
        );
    } else if (M >= 8) {
        // Larger batch: use simple thread-per-output kernel (good occupancy)
        int threads_per_block = min(256, (int)N);
        dim3 block(threads_per_block);
        dim3 grid(M, (N + threads_per_block - 1) / threads_per_block);
        q4_0_gemm_simple<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, (int)M, (int)N, (int)K
        );
    } else {
        // Medium batch: use warp-based kernel for better utilization
        dim3 block(32);
        dim3 grid(M, N);
        q4_0_gemm_warp<<<grid, block>>>(
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
    m.def("forward", &forward, "Q4_0 GEMM Forward - DeepSeek-V2 Attention Output");
}
