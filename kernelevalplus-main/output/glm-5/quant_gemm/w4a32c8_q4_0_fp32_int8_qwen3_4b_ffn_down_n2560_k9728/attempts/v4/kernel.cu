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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v4 - DP4A + Dynamic Quantization)
 *
 * Qwen3-4B FFN Down projection
 * - N = 2560 (output features)
 * - K = 9728 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * W4A32C8 uses dynamic quantization:
 * - Weights: Q4_0 (4-bit packed with scale d4_0)
 * - Activation: FP32, dynamically quantized to Q8_1 per block in kernel
 * - Formula: result = d4_0 * (d8_1 * sumi - 8 * s8_1)
 *
 * This approach enables integer arithmetic (DP4A) for maximum performance.
 */

#define QK 32

// DP4A dot product: computes a[i]*b[i] for i=0..3 and adds to c
__device__ __inline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    // Fallback for older architectures
    int8_t *va = (int8_t*)&a;
    int8_t *vb = (int8_t*)&b;
    return c + va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2] + va[3]*vb[3];
#endif
}

// Read FP16 value as float
__device__ __inline__ float read_fp16(const uint8_t* p) {
    uint16_t u = p[0] | ((uint16_t)p[1] << 8);
    union { uint16_t u16; __half f16; } un;
    un.u16 = u;
    return __half2float(un.f16);
}

// Compute dot product of one Q4_0 block with dynamically quantized FP32
// Returns: d4_0 * (d8_1 * sumi - 8 * s8_1)
__device__ __inline__ float q4_0_fp32_dot_block(
    const uint8_t* __restrict__ w_ptr,  // Q4_0 weight block [18 bytes]
    const float* __restrict__ a_ptr        // FP32 activation block [32 values]
) {
    // Read Q4_0 weight scale
    float d4_0 = read_fp16(w_ptr);

    // Dynamically quantize FP32 activation to Q8_1 per block
    // Compute max and sum for quantization
    float amax = 0.0f;
    float asum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        float v = a_ptr[i];
        asum += v;
        float av = fabsf(v);
        amax = fmaxf(amax, av);
    }

    // Compute Q8_1 scale (d8_1)
    float d8_1 = amax / 127.0f;
    if (d8_1 < 1e-10f) d8_1 = 1.0f;
    float inv_d = 1.0f / d8_1;

    // Quantize FP32 to INT8 (Q8_1 values)
    int8_t a_qs[QK];
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        int v = __float2int_rn(a_ptr[i] * inv_d);
        a_qs[i] = (int8_t)max(-128, min(127, v));
    }

    // Compute integer dot product sumi = sum(w_qs[i] * a_qs[i])
    // Q4_0 unpacking: low nibbles first (0-15), then high nibbles (16-31)
    const uint8_t* w_qs_packed = w_ptr + 2;  // Skip scale

    int sumi = 0;

    // Process 16 bytes of packed Q4_0, each contains 2 values
    // Use DP4A for efficient 4-way dot products
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Pack low nibbles of 4 consecutive bytes into an int
        int8_t tl[4], th[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t b = w_qs_packed[i*4 + j];
            tl[j] = (int8_t)(b & 0x0F);
            th[j] = (int8_t)((b >> 4) & 0x0F);
        }

        // DP4A: accumulate a[i]*b[i] for i=0..3
        sumi = dp4a(*reinterpret_cast<int*>(tl), *reinterpret_cast<int*>(&a_qs[i*4]), sumi);
        sumi = dp4a(*reinterpret_cast<int*>(th), *reinterpret_cast<int*>(&a_qs[16+i*4]), sumi);
    }

    // Apply compensation formula: d4_0 * (d8_1 * sumi - 8 * s8_1)
    return d4_0 * (d8_1 * (float)sumi - 8.0f * asum);
}

// Small M kernel: one thread per output, direct global memory reads
__global__ void quant_gemm_q4_0_fp32_kernel_small(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const int num_blocks = K / QK;
    float acc = 0.0f;

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row_base = weight + n_idx * num_blocks * 18;

    // Process all K blocks
    int kb = 0;
    for (; kb + 1 < num_blocks; kb += 2) {
        // Process 2 blocks for better ILP
        acc += q4_0_fp32_dot_block(weight_row_base + kb * 18, act_row + kb * QK);
        acc += q4_0_fp32_dot_block(weight_row_base + (kb + 1) * 18, act_row + (kb + 1) * QK);
    }

    // Handle remaining block
    if (kb < num_blocks) {
        acc += q4_0_fp32_dot_block(weight_row_base + kb * 18, act_row + kb * QK);
    }

    output[m_idx * N + n_idx] = acc;
}

// Large M kernel: shared memory tiling
__global__ void quant_gemm_q4_0_fp32_kernel_large(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.y;
    const int n_base = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_base >= N) return;

    const int num_blocks = K / QK;
    float acc = 0.0f;

    // Shared memory for activation tiling
    __shared__ float s_act[QK * 4];  // Tile 4 blocks

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row_base = weight + n_base * num_blocks * 18;

    // Process in chunks of 4 blocks
    int kb = 0;
    const int chunk_size = 4;

    for (; kb + chunk_size <= num_blocks; kb += chunk_size) {
        // Cooperatively load activation blocks into shared memory
        int tidx = threadIdx.x;
        if (tidx < QK) {
            #pragma unroll
            for (int c = 0; c < chunk_size; c++) {
                s_act[c * QK + tidx] = act_row[(kb + c) * QK + tidx];
            }
        }
        __syncthreads();

        // Process all 4 blocks from shared memory
        #pragma unroll
        for (int c = 0; c < chunk_size; c++) {
            const float* a_ptr = s_act + c * QK;
            const uint8_t* w_ptr = weight_row_base + (kb + c) * 18;
            acc += q4_0_fp32_dot_block(w_ptr, a_ptr);
        }

        __syncthreads();
    }

    // Handle remaining blocks
    for (; kb < num_blocks; kb++) {
        int tidx = threadIdx.x;
        if (tidx < QK) {
            s_act[tidx] = act_row[kb * QK + tidx];
        }
        __syncthreads();

        acc += q4_0_fp32_dot_block(weight_row_base + kb * 18, s_act);

        __syncthreads();
    }

    output[m_idx * N + n_base] = acc;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    auto weight_contig = weight.contiguous();
    auto act_contig = activation.contiguous();

    int threads_per_block = 256;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = M;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    // Strategy dispatch based on batch size
    if (M < 16) {
        quant_gemm_q4_0_fp32_kernel_small<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        quant_gemm_q4_0_fp32_kernel_large<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V4 - DP4A + Dynamic Quantization)");
}
