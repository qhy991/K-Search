#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * W4A32C8 Q4_0 × FP32_INT8 Quantized GEMM Kernel v2
 *
 * DeepSeek-V2 MoE Routing Down Projection
 * - N = 1536 (output features)
 * - K = 5120 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Q4_0 format (18 bytes per block):
 *   - scale: FP16 (2 bytes)
 *   - qs: packed 4-bit values (16 bytes)
 * Q4_0 encoding: q = round(val / scale + 8), q ∈ [0, 15]
 * Q4_0 decoding: val = scale × (q - 8)
 *
 * Computation (llama.cpp Q4_0 × Q8_1 style):
 * 1. Dynamically quantize FP32 activation to INT8 per-block
 * 2. Compute INT8 dot product with Q4_0 weights
 * 3. Apply compensation: result = d_w * (d_a * sumi - 8 * a_sum)
 *
 * Strategy: Based on Roofline analysis
 * - Small M (M < 8): Direct memory access (memory-bound, minimize overhead)
 * - Large M (M >= 8): Shared memory tiling (compute-bound, maximize throughput)
 */

#define Q4_0_BLOCK_SIZE 32

// Union for safe FP16 to FP32 conversion
union half_union {
    uint16_t u16;
    __half h16;
};

// ========== Small M kernel: direct access, optimized for memory bandwidth ==========
__global__ void quant_gemm_small_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    float acc = 0.0f;

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_idx * num_blocks * 18;

    // Process all blocks in K dimension
    for (int kb = 0; kb < num_blocks; kb++) {
        const float* act_block = act_row + kb * Q4_0_BLOCK_SIZE;
        const uint8_t* weight_block = weight_row + kb * 18;

        // === Load weight scale (FP16) ===
        half_union hu;
        hu.u16 = *reinterpret_cast<const uint16_t*>(weight_block);
        const float d_w = __half2float(hu.h16);
        const uint8_t* w_qs = weight_block + 2;

        // === Dynamic quantization of FP32 activation to INT8 ===
        // Find activation scale for this block
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_max = fmaxf(a_max, fabsf(act_block[i]));
        }
        const float d_a = fmaxf(a_max / 127.0f, 1e-7f);

        // Compute activation sum for Q8_1 style compensation
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_sum += act_block[i];
        }

        // === INT8 dot product with Q4_0 weights ===
        // Q4_0 stores values 0-15, use directly as INT8 (subtracting 8 is compensated)
        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;           // values 0-15
            const int w_high = (byte_val >> 4) & 0x0F;   // values 0-15

            // Quantize activation to int8
            const int a_low = __float2int_rn(act_block[i] / d_a);
            const int a_high = __float2int_rn(act_block[i + 16] / d_a);

            sumi += w_low * a_low;
            sumi += w_high * a_high;
        }

        // === Apply llama.cpp compensation formula ===
        // result = d_w * (d_a * sumi - 8 * a_sum)
        acc += d_w * (d_a * sumi - 8.0f * a_sum);
    }

    output[m_idx * N + n_idx] = acc;
}

// ========== Large M kernel: shared memory tiling ==========
__global__ void quant_gemm_large_m(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_base = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_base >= N) return;

    const int num_blocks = K / Q4_0_BLOCK_SIZE;
    float acc = 0.0f;

    // Shared memory for activation block (32 values)
    __shared__ float act_shared[Q4_0_BLOCK_SIZE];

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row = weight + n_base * num_blocks * 18;

    // Iterate over K dimension in blocks of 32
    for (int kb = 0; kb < num_blocks; kb++) {
        // Load activation block into shared memory
        if (threadIdx.x < Q4_0_BLOCK_SIZE) {
            act_shared[threadIdx.x] = act_row[kb * Q4_0_BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();

        // Load weight block
        const uint8_t* weight_block = weight_row + kb * 18;

        half_union hu;
        hu.u16 = *reinterpret_cast<const uint16_t*>(weight_block);
        const float d_w = __half2float(hu.h16);
        const uint8_t* w_qs = weight_block + 2;

        // === Dynamic quantization of FP32 activation to INT8 ===
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_max = fmaxf(a_max, fabsf(act_shared[i]));
        }
        const float d_a = fmaxf(a_max / 127.0f, 1e-7f);

        // Compute activation sum
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_sum += act_shared[i];
        }

        // === INT8 dot product with Q4_0 weights ===
        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;

            const int a_low = __float2int_rn(act_shared[i] / d_a);
            const int a_high = __float2int_rn(act_shared[i + 16] / d_a);

            sumi += w_low * a_low;
            sumi += w_high * a_high;
        }

        // Apply compensation formula
        acc += d_w * (d_a * sumi - 8.0f * a_sum);

        __syncthreads();
    }

    output[m_idx * N + n_base] = acc;
}

// ========== Host function with strategy dispatch ==========
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    auto weight_contig = weight.contiguous();
    auto act_contig = activation.contiguous();

    // Configure thread block
    // 256 threads per block is optimal for RTX 4090
    int threads_per_block = 256;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    // Strategy dispatch based on M (Roofline analysis)
    // Small M (M < 8): Direct access - memory-bound, minimize synchronization overhead
    // Large M (M >= 8): Shared memory - compute-bound, maximize throughput
    if (M < 8) {
        quant_gemm_small_m<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        quant_gemm_large_m<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32_INT8 Quantized GEMM v2");
}
