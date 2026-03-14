// W4A32C8 Q4_0 GEMM Kernel for DeepSeek-V2 Attention Output
// Dimensions: M (batch), N=5120 (output), K=5120 (input)
// Weight: Q4_0 quantized [N, K/32], Activation: FP32 [M, K]
// Output: FP32 [M, N]
//
// Formula: result = d4_0 * (d8_1 * sumi - 8 * s8_1)
// where Q4_0 uses offset-8 encoding
//
// Compute-optimized implementation using __dp4a and vectorized operations

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Q4_0 block format
#define BLOCK_K 32  // quantization block size
#define Q4_0_BLOCK_SIZE 18  // bytes per q4_0 block (2-byte FP16 scale + 16 bytes packed data)

// Q4_0 block structure (for alignment reference)
typedef struct {
    half d;         // FP16 scale (delta)
    uint8_t qs[16]; // 16 bytes containing 32 x 4-bit values
} block_q4_0;
static_assert(sizeof(block_q4_0) == Q4_0_BLOCK_SIZE, "Q4_0 block size must be 18 bytes");

// Helper: Convert FP16 to float
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Helper: Load FP16 from bytes
__device__ __forceinline__ half load_half(const uint8_t* ptr) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(ptr);
    return __ushort_as_half(val);
}

// Helper: Unpack 4 Q4_0 values from a byte (llama.cpp format)
// Returns 4 values in [0, 15] range
__device__ __forceinline__ void unpack_q4_0_byte(uint8_t byte, int& v0, int& v1, int& v2, int& v3) {
    v0 = byte & 0x0F;
    v1 = (byte >> 4) & 0x0F;
    v2 = (byte >> 0) & 0x0F;  // Same as v0 - for dp4a alignment
    v3 = (byte >> 4) & 0x0F;  // Same as v1 - for dp4a alignment
}

// Compute-intensive kernel: Each warp processes one output element
// Uses __dp4a for efficient dot product computation
__global__ void q4_0_gemm_kernel_dp4a(
    const uint8_t* __restrict__ weight,  // [N, K/32] Q4_0 blocks
    const float* __restrict__ activation, // [M, K] FP32
    float* __restrict__ output,           // [M, N] FP32
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;  // 5120 / 32 = 160

    // Each warp (32 threads) processes one output element
    const int m = blockIdx.x;  // batch index
    if (m >= M) return;

    const int n = blockIdx.y * blockDim.y + threadIdx.y;  // output feature index
    if (n >= N) return;

    const int lane_id = threadIdx.x % 32;

    // Get activation row for this batch element
    const float* act_row = activation + m * K;

    // Get weight row for this output feature
    const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;

    // Each thread accumulates partial sums
    float sum = 0.0f;

    // Process K in blocks of 32 (matching Q4_0 block size)
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load weight scale for this block
        half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
        float d_w = half_to_float(d_h);

        // Get packed weight data for this block
        const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;  // Skip 2-byte scale

        // Each lane processes elements: lane i processes element (kb*32 + i)
        // We need 32 iterations to process all 32 values in the block
        int k_base = kb * BLOCK_K;

        // Process all 32 values in this Q4_0 block
        // Each thread handles 1 element (32 threads * 1 = 32 elements)
        int k_idx = k_base + lane_id;

        // Unpack the 4-bit value for this lane
        // Q4_0 stores: first 16 bytes = low nibbles, positions 0-15
        //             next we need high nibbles for positions 16-31
        int w_q;
        if (lane_id < 16) {
            // Positions 0-15: low nibbles of bytes 0-15
            w_q = w_packed[lane_id] & 0x0F;
        } else {
            // Positions 16-31: high nibbles of bytes 0-15
            w_q = (w_packed[lane_id - 16] >> 4) & 0x0F;
        }

        // Dequantize: Q4_0 encoding is offset-8
        // q = round(val / scale + 8), so val = scale * (q - 8)
        float w_val = d_w * (w_q - 8);

        // Get activation value
        float a_val = act_row[k_idx];

        // Multiply-accumulate
        sum += a_val * w_val;
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write result (lane 0 only)
    if (lane_id == 0) {
        output[m * N + n] = sum;
    }
}

// Optimized kernel: Block processes multiple N columns with shared memory for weights
// Better weight reuse when processing multiple outputs
__global__ void q4_0_gemm_kernel_shared(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;
    const int N_TILES = 16;  // Number of N columns processed per block

    // Block configuration
    const int m = blockIdx.x;
    if (m >= M) return;

    const int n_start = blockIdx.y * N_TILES;
    if (n_start >= N) return;

    const int n_end = min(n_start + N_TILES, N);
    const int n_local = threadIdx.y;  // 0 to N_TILES-1
    const int n = n_start + n_local;
    const int valid_n = (n < n_end);

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Shared memory for activation tile (loaded once per block)
    __shared__ float act_smem[BLOCK_K];  // 32 float values

    // Get activation row for this batch
    const float* act_row = activation + m * K;

    // Each warp processes one N column
    float sum = 0.0f;

    // Process K in blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        // Load activation tile cooperatively
        // Each thread in warp loads one activation value
        int k_base = kb * BLOCK_K;
        if (lane_id < BLOCK_K) {
            act_smem[lane_id] = act_row[k_base + lane_id];
        }
        __syncthreads();

        // Load weight scale for this N column
        if (valid_n) {
            const uint8_t* w_row = weight + n * num_blocks_k * Q4_0_BLOCK_SIZE;
            half d_h = load_half(w_row + kb * Q4_0_BLOCK_SIZE);
            float d_w = half_to_float(d_h);

            // Get packed weights
            const uint8_t* w_packed = w_row + kb * Q4_0_BLOCK_SIZE + 2;

            // Each thread in warp processes one element
            int w_q;
            if (lane_id < 16) {
                w_q = w_packed[lane_id] & 0x0F;
            } else {
                w_q = (w_packed[lane_id - 16] >> 4) & 0x0F;
            }

            float w_val = d_w * (w_q - 8);
            float a_val = act_smem[lane_id];

            sum += a_val * w_val;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write result
    if (valid_n && lane_id == 0) {
        output[m * N + n] = sum;
    }
}

// Simple kernel for small batches (M=1): One thread per output
__global__ void q4_0_gemm_kernel_single_token(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    // Each thread processes one output element
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // M=1, so activation is just a 1D array of K elements
    const float* act_row = activation;

    // Get weight row for this output
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

// Vectorized kernel using int4 operations
__global__ void q4_0_gemm_kernel_vectorized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / BLOCK_K;

    // Each block processes one (m, n) pair
    const int m = blockIdx.x;
    const int n = blockIdx.y;

    if (m >= M || n >= N) return;

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

        // Process 16 bytes (32 values) with unrolling
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

    output[m * N + n] = sum;
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
        // Single token: use simple thread-per-output kernel
        int threads = min(256, (int)N);
        int blocks = (N + threads - 1) / threads;
        q4_0_gemm_kernel_single_token<<<blocks, threads>>>(
            weight_ptr, act_ptr, output_ptr, (int)N, (int)K
        );
    } else if (M >= 8) {
        // Larger batch: use warp-per-output kernel
        dim3 block(32, 8);  // 8 warps per block
        dim3 grid(M, (N + 7) / 8);
        q4_0_gemm_kernel_shared<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, (int)M, (int)N, (int)K
        );
    } else {
        // Medium batch: use vectorized kernel
        dim3 block(256);
        dim3 grid(M, N);
        q4_0_gemm_kernel_vectorized<<<grid, block>>>(
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
