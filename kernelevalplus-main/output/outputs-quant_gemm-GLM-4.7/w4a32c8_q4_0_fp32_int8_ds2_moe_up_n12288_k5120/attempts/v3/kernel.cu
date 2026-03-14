#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure: 18 bytes per block
typedef struct {
    uint16_t d;     // scale (FP16)
    uint8_t qs[16]; // packed 4-bit values
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size must be 18 bytes");

// Device function to convert FP16 to FP32
__device__ __forceinline__ float fp16_to_fp32(uint16_t fp16_val) {
    union { uint16_t u16; __half fp16; } un;
    un.u16 = fp16_val;
    return __half2float(un.fp16);
}

// Optimized dot product using vectorized loads and __ldg
__device__ __forceinline__ float dot_q4_0_fp32_optimized(const block_q4_0* w_block, const float* act_block) {
    float d_w = fp16_to_fp32(w_block->d);

    // Use __ldg for read-only cache hint
    float sum = 0.0f;
    const uint8_t* qs = w_block->qs;

    // Process 8 values at a time for better instruction pipelining
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t packed0 = qs[i];
        uint8_t packed1 = qs[i + 8];

        // Unpack both bytes
        int q_low0 = packed0 & 0x0F;
        int q_high0 = (packed0 >> 4) & 0x0F;
        int q_low1 = packed1 & 0x0F;
        int q_high1 = (packed1 >> 4) & 0x0F;

        // Load activations with __ldg
        sum += d_w * (q_low0 - 8) * __ldg(&act_block[i]);
        sum += d_w * (q_high0 - 8) * __ldg(&act_block[i + 16]);
        sum += d_w * (q_low1 - 8) * __ldg(&act_block[i + 8]);
        sum += d_w * (q_high1 - 8) * __ldg(&act_block[i + 24]);
    }

    return sum;
}

// Kernel for small M: Optimized for memory bandwidth
// Each thread handles multiple N elements for better coalescing
__global__ void w4a32c8_q4_0_kernel_small_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    // Each thread processes multiple N values for better memory coalescing
    const int N_PER_THREAD = 4;
    int base_col = (blockIdx.x * blockDim.x + threadIdx.x) * N_PER_THREAD;
    int row = blockIdx.y;

    if (row >= M || base_col >= N) return;

    float sum[N_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; i++) {
        sum[i] = 0.0f;
    }

    const float* act_row = activation + row * K;

    // Process all blocks for all assigned columns
    int valid_cols = min(N_PER_THREAD, N - base_col);

    for (int block = 0; block < num_blocks; block++) {
        const float* act_block = act_row + block * 32;

        // Pre-load activation block into registers (vectorized)
        float a_vals[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_block[i * 4]);
            a_vals[i * 4 + 0] = a4.x;
            a_vals[i * 4 + 1] = a4.y;
            a_vals[i * 4 + 2] = a4.z;
            a_vals[i * 4 + 3] = a4.w;
        }

        // Compute for each valid column
        for (int c = 0; c < valid_cols; c++) {
            int col = base_col + c;
            const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
            const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);
            const block_q4_0* w_block = &w_col[block];

            float d_w = fp16_to_fp32(w_block->d);
            const uint8_t* qs = w_block->qs;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = qs[i];
                int q_low = packed & 0x0F;
                int q_high = (packed >> 4) & 0x0F;

                sum[c] += d_w * (q_low - 8) * a_vals[i];
                sum[c] += d_w * (q_high - 8) * a_vals[i + 16];
            }
        }
    }

    // Write results
    for (int c = 0; c < valid_cols; c++) {
        output[row * N + base_col + c] = sum[c];
    }
}

// Kernel for large M: Optimized with shared memory and vectorized loads
__global__ void w4a32c8_q4_0_kernel_large_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    // Each thread processes 2 N values
    const int N_PER_THREAD = 2;
    int base_col = (blockIdx.x * blockDim.x + threadIdx.x) * N_PER_THREAD;
    int row = blockIdx.y;

    if (row >= M || base_col >= N) return;

    float sum[N_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; i++) {
        sum[i] = 0.0f;
    }

    const float* act_row = activation + row * K;
    int valid_cols = min(N_PER_THREAD, N - base_col);

    // Shared memory for activation tiles
    __shared__ float s_act[32];

    for (int block = 0; block < num_blocks; block++) {
        // Load activation block into shared memory (coalesced)
        if (threadIdx.x < 32) {
            int load_idx = threadIdx.x;
            s_act[load_idx] = act_row[block * 32 + load_idx];
        }
        __syncthreads();

        // Compute for each valid column
        for (int c = 0; c < valid_cols; c++) {
            int col = base_col + c;
            const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
            const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);
            const block_q4_0* w_block = &w_col[block];

            float d_w = fp16_to_fp32(w_block->d);
            const uint8_t* qs = w_block->qs;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = qs[i];
                int q_low = packed & 0x0F;
                int q_high = (packed >> 4) & 0x0F;

                sum[c] += d_w * (q_low - 8) * s_act[i];
                sum[c] += d_w * (q_high - 8) * s_act[i + 16];
            }
        }

        __syncthreads();
    }

    // Write results
    for (int c = 0; c < valid_cols; c++) {
        output[row * N + base_col + c] = sum[c];
    }
}

// Host function to dispatch kernels based on M
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* activation_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if (M <= 8) {
        // Small batch: process 4 N values per thread for better coalescing
        const int THREADS_PER_BLOCK = 256;  // Each thread handles 4 N values
        const int N_PER_BLOCK = THREADS_PER_BLOCK * 4;
        const int num_blocks_x = (N + N_PER_BLOCK - 1) / N_PER_BLOCK;

        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(num_blocks_x, M);

        w4a32c8_q4_0_kernel_small_batch<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else {
        // Large batch: process 2 N values per thread with shared memory
        const int THREADS_PER_BLOCK = 256;
        const int N_PER_BLOCK = THREADS_PER_BLOCK * 2;
        const int num_blocks_x = (N + N_PER_BLOCK - 1) / N_PER_BLOCK;

        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(num_blocks_x, M);

        w4a32c8_q4_0_kernel_large_batch<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
