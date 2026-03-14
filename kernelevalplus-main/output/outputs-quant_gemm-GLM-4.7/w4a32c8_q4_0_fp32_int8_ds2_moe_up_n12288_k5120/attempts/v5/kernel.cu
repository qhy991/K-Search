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

// Optimized dot product using __ldg for cache
__device__ __forceinline__ float dot_q4_0_fp32_ldg(const block_q4_0* w_block, const float* act_block) {
    float d_w = fp16_to_fp32(w_block->d);
    const uint8_t* qs = w_block->qs;

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t packed = qs[i];
        int q_low = packed & 0x0F;
        int q_high = (packed >> 4) & 0x0F;

        sum += d_w * (q_low - 8) * __ldg(&act_block[i]);
        sum += d_w * (q_high - 8) * __ldg(&act_block[i + 16]);
    }

    return sum;
}

// Simple kernel: one thread per output element
__global__ void w4a32c8_q4_0_kernel_simple(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const float* act_row = activation + row * K;

    const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
    const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);

    for (int block = 0; block < num_blocks; block++) {
        const float* act_block = act_row + block * 32;
        sum += dot_q4_0_fp32_ldg(&w_col[block], act_block);
    }

    output[row * N + col] = sum;
}

// Kernel with shared memory for activation tiles
__global__ void w4a32c8_q4_0_kernel_shared(
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
        // Load activation block into shared memory
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

// Kernel with vectorized loads for better memory bandwidth
__global__ void w4a32c8_q4_0_kernel_vectorized(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const float* act_row = activation + row * K;

    const uint8_t* w_row_bytes = weight + col * num_blocks * 18;
    const block_q4_0* w_col = reinterpret_cast<const block_q4_0*>(w_row_bytes);

    for (int block = 0; block < num_blocks; block++) {
        const float* act_block = act_row + block * 32;
        const block_q4_0* w_block = &w_col[block];

        float d_w = fp16_to_fp32(w_block->d);
        const uint8_t* qs = w_block->qs;

        // Vectorized loads for activation
        float a_vals[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 a = *reinterpret_cast<const float4*>(&act_block[i * 4]);
            a_vals[i * 4 + 0] = a.x;
            a_vals[i * 4 + 1] = a.y;
            a_vals[i * 4 + 2] = a.z;
            a_vals[i * 4 + 3] = a.w;
        }

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = qs[i];
            int q_low = packed & 0x0F;
            int q_high = (packed >> 4) & 0x0F;

            sum += d_w * (q_low - 8) * a_vals[i];
            sum += d_w * (q_high - 8) * a_vals[i + 16];
        }
    }

    output[row * N + col] = sum;
}

// Host function with adaptive kernel selection
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

    const int threads_per_block = 256;
    const int num_blocks_col = (N + threads_per_block - 1) / threads_per_block;

    dim3 block(threads_per_block);
    dim3 grid(num_blocks_col, M);

    if (M == 1) {
        // Single token: use __ldg for cache optimization
        w4a32c8_q4_0_kernel_simple<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else if (M <= 8) {
        // Small batch: use vectorized loads
        w4a32c8_q4_0_kernel_vectorized<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else {
        // Large batch: use shared memory
        w4a32c8_q4_0_kernel_shared<<<grid, block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
