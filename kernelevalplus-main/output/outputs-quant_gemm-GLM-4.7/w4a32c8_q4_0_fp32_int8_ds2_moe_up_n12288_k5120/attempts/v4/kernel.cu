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

// Warp-level dot product: each lane processes 2 values
__device__ __forceinline__ float warp_dot_q4_0_fp32(
    const block_q4_0* w_block,
    const float* act_block,
    int lane_id
) {
    float d_w = fp16_to_fp32(w_block->d);
    const uint8_t* qs = w_block->qs;

    // Each lane processes 2 packed bytes (4 values)
    int idx = lane_id % 16;
    uint8_t packed0 = qs[idx];
    uint8_t packed1 = qs[idx];

    int q_low0 = packed0 & 0x0F;
    int q_high0 = (packed0 >> 4) & 0x0F;
    int q_low1 = packed1 & 0x0F;
    int q_high1 = (packed1 >> 4) & 0x0F;

    float sum = d_w * (q_low0 - 8) * act_block[idx];
    sum += d_w * (q_high0 - 8) * act_block[idx + 16];

    return sum;
}

// Kernel for small M (1-8): Warp-level processing for memory bandwidth
__global__ void w4a32c8_q4_0_kernel_small_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const int WARP_SIZE = 32;

    // Warp layout
    int warp_id = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int total_warps = gridDim.x * blockDim.x / WARP_SIZE;

    // Each warp processes multiple (row, col) pairs
    for (int idx = warp_id; idx < M * N; idx += total_warps) {
        int row = idx / N;
        int col = idx % N;

        float sum = 0.0f;
        const float* act_row = activation + row * K;

        const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
        const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);

        for (int block = 0; block < num_blocks; block++) {
            const float* act_block = act_row + block * 32;
            const block_q4_0* w_block = &w_col[block];

            float d_w = fp16_to_fp32(w_block->d);
            const uint8_t* qs = w_block->qs;

            // Each lane processes 2 values
            int base = lane_id % 16;
            uint8_t packed = qs[base];

            int q_low = packed & 0x0F;
            int q_high = (packed >> 4) & 0x0F;

            float partial = d_w * (q_low - 8) * act_block[base];
            partial += d_w * (q_high - 8) * act_block[base + 16];

            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }

            if (lane_id == 0) {
                sum += partial;
            }
        }

        if (lane_id == 0) {
            output[row * N + col] = sum;
        }
    }
}

// Kernel for larger M: Optimized with shared memory
__global__ void w4a32c8_q4_0_kernel_large_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    __shared__ float s_act[32];

    float sum = 0.0f;
    const float* act_row = activation + row * K;

    const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
    const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);

    for (int block = 0; block < num_blocks; block++) {
        // Load activation into shared memory
        if (threadIdx.x < 32) {
            s_act[threadIdx.x] = act_row[block * 32 + threadIdx.x];
        }
        __syncthreads();

        float d_w = fp16_to_fp32(w_col[block].d);
        const uint8_t* qs = w_col[block].qs;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            int q_low = packed & 0x0F;
            int q_high = (packed >> 4) & 0x0F;

            sum += d_w * (q_low - 8) * s_act[i];
            sum += d_w * (q_high - 8) * s_act[i + 16];
        }

        __syncthreads();
    }

    output[row * N + col] = sum;
}

// Host function
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
        // Small batch: use warp-level kernel
        const int threads_per_block = 256;
        const int num_blocks = min(128, (M * N + 7) / 8);

        w4a32c8_q4_0_kernel_small_batch<<<num_blocks, threads_per_block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else {
        // Large batch: use shared memory kernel
        const int threads_per_block = 256;
        const int num_blocks_col = (N + threads_per_block - 1) / threads_per_block;

        dim3 block(threads_per_block);
        dim3 grid(num_blocks_col, M);

        w4a32c8_q4_0_kernel_large_batch<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
