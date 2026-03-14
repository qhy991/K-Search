#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Q4_1 block structure: 20 bytes per 32 elements
// d (FP16 scale): 2 bytes, m (FP16 min): 2 bytes, qs (packed 4-bit values): 16 bytes
typedef struct {
    uint16_t d;  // FP16 scale
    uint16_t m;  // FP16 min
    uint8_t qs[16];  // packed 4-bit values (32 values total)
} block_q4_1;
static_assert(sizeof(block_q4_1) == 20, "Q4_1 block size must be 20 bytes");

// Device function to convert FP16 to FP32
__device__ __inline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Path 1: Simple vectorized loads for small batches (memory-bound)
// Optimized for M=1 to M=8 where OI is low
__global__ void q4_1_fp32_gemm_small_batch(
    const uint8_t* __restrict__ weight_q4_1,
    const float* __restrict__ activation_fp32,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;
    const int m = blockIdx.x;  // batch index
    const int n = blockIdx.y * blockDim.x + threadIdx.x;  // output element index

    if (n >= N) return;

    const float* act_row = activation_fp32 + m * K;
    float acc = 0.0f;

    // Process K in blocks of 32
    for (int kb = 0; kb < K_BLOCKS; kb++) {
        int k_base = kb * 32;

        // Load and compute statistics for activation quantization
        float act_max_abs = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float val = act_row[k_base + i];
            act_max_abs = fmaxf(act_max_abs, fabsf(val));
            act_sum += val;
        }

        // Compute activation scale for Q8_1-style quantization
        float d_a = fmaxf(act_max_abs / 127.0f, 1e-6f);

        // Load weight block
        const block_q4_1* w_block = reinterpret_cast<const block_q4_1*>(
            weight_q4_1 + (n * K_BLOCKS + kb) * sizeof(block_q4_1)
        );

        float d_w = fp16_to_fp32(w_block->d);
        float m_w = fp16_to_fp32(w_block->m);

        // Compute dot product with unpacking
        int sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_block->qs[i];
            int q_low = packed & 0x0F;
            int q_high = (packed >> 4) & 0x0F;

            float a_low = act_row[k_base + i];
            float a_high = act_row[k_base + i + 16];

            int q8_low = __float2int_rn(a_low / d_a);
            int q8_high = __float2int_rn(a_high / d_a);

            sumi += q_low * q8_low;
            sumi += q_high * q8_high;
        }

        // Apply dequantization: result = d_w * d_a * sumi + m_w * act_sum
        acc += d_w * d_a * (float)sumi + m_w * act_sum;
    }

    output[m * N + n] = acc;
}

// Path 2: Shared memory tiling for large batches (compute-bound)
// Each thread handles one output element (n), and we tile across M
__global__ void q4_1_fp32_gemm_large_batch(
    const uint8_t* __restrict__ weight_q4_1,
    const float* __restrict__ activation_fp32,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;

    // Each thread computes one output element (n) across all M
    const int m_start = blockIdx.y;
    const int m_stride = gridDim.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N) return;

    // Process batch rows in strided fashion
    for (int m = m_start; m < M; m += m_stride) {
        const float* act_row = activation_fp32 + m * K;
        float acc = 0.0f;

        for (int kb = 0; kb < K_BLOCKS; kb++) {
            int k_base = kb * 32;

            // Compute activation statistics
            float act_max_abs = 0.0f;
            float act_sum = 0.0f;

            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float val = act_row[k_base + i];
                act_max_abs = fmaxf(act_max_abs, fabsf(val));
                act_sum += val;
            }

            float d_a = fmaxf(act_max_abs / 127.0f, 1e-6f);

            // Load weight block
            const block_q4_1* w_block = reinterpret_cast<const block_q4_1*>(
                weight_q4_1 + (n * K_BLOCKS + kb) * sizeof(block_q4_1)
            );

            float d_w = fp16_to_fp32(w_block->d);
            float m_w = fp16_to_fp32(w_block->m);

            // Compute dot product
            int sumi = 0;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = w_block->qs[i];
                int q_low = packed & 0x0F;
                int q_high = (packed >> 4) & 0x0F;

                float a_low = act_row[k_base + i];
                float a_high = act_row[k_base + i + 16];

                int q8_low = __float2int_rn(a_low / d_a);
                int q8_high = __float2int_rn(a_high / d_a);

                sumi += q_low * q8_low;
                sumi += q_high * q8_high;
            }

            acc += d_w * d_a * (float)sumi + m_w * act_sum;
        }

        output[m * N + n] = acc;
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.scalar_type() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Strategy dispatch based on batch size (Roofline analysis)
    // Small M (1-8): Memory-bound -> use simple kernel with vectorized loads
    // Large M (>=16): Compute-bound -> use tiled kernel for better compute utilization
    if (M <= 8) {
        // Path 1: Simple kernel for memory-bound cases
        int TILE_N = (M == 1) ? 128 : 256;  // Adjust thread count based on M

        dim3 block(TILE_N);
        dim3 grid(M, (N + TILE_N - 1) / TILE_N);

        q4_1_fp32_gemm_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Path 2: Tiled kernel for compute-bound cases
        // Each thread computes one output element (n), and we tile across M
        int THREADS_X = 128;  // Threads per block in x dimension (N dimension)
        int GRID_Y = M;      // One block per batch row for simplicity

        int grid_x = (N + THREADS_X - 1) / THREADS_X;

        dim3 block(THREADS_X);
        dim3 grid(grid_x, GRID_Y);

        q4_1_fp32_gemm_large_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "CUDA kernel synchronization failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_1 FP32 GEMM with dual-path dispatch");
}
