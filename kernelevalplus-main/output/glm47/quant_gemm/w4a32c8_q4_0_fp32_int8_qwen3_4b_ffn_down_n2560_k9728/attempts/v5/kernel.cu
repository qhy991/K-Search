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
 * W4A32C8 Q4_0 × FP32 Quantized GEMM Kernel (v5 - Hybrid FP32/DP4A)
 *
 * Qwen3-4B FFN Down projection
 * - N = 2560 (output features)
 * - K = 9728 (input features, must be multiple of 32)
 * - M = variable (batch size)
 *
 * Hybrid approach:
 * - For M < 32: Memory-bound, use simple FP32 dequantization
 * - For M >= 32: Compute-bound, use DP4A with dynamic quantization
 */

#define QK 32

// DP4A dot product
__device__ __inline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int8_t *va = (int8_t*)&a;
    int8_t *vb = (int8_t*)&b;
    return c + va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2] + va[3]*vb[3];
#endif
}

// Read FP16 as float
__device__ __inline__ float read_fp16(const uint8_t* p) {
    uint16_t u = p[0] | ((uint16_t)p[1] << 8);
    union { uint16_t u16; __half f16; } un;
    un.u16 = u;
    return __half2float(un.f16);
}

// FP32 version: simple dequantization and multiply
__device__ __inline__ float q4_0_fp32_dot_fp32(
    const uint8_t* __restrict__ w_ptr,
    const float* __restrict__ a_ptr
) {
    float dw = read_fp16(w_ptr);
    const uint8_t* w_packed = w_ptr + 2;
    float acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t b = w_packed[i];
        int w_low = (int)(b & 0x0F) - 8;
        int w_high = (int)((b >> 4) & 0x0F) - 8;
        acc += a_ptr[i] * (dw * w_low);
        acc += a_ptr[i + 16] * (dw * w_high);
    }
    return acc;
}

// DP4A version: dynamic quantization
__device__ __inline__ float q4_0_fp32_dot_dp4a(
    const uint8_t* __restrict__ w_ptr,
    const float* __restrict__ a_ptr
) {
    float d4_0 = read_fp16(w_ptr);

    float amax = 0.0f;
    float asum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        float v = a_ptr[i];
        asum += v;
        float av = fabsf(v);
        amax = fmaxf(amax, av);
    }

    float d8_1 = amax / 127.0f;
    if (d8_1 < 1e-10f) d8_1 = 1.0f;
    float inv_d = 1.0f / d8_1;

    int8_t a_qs[QK];
    #pragma unroll
    for (int i = 0; i < QK; i++) {
        int v = __float2int_rn(a_ptr[i] * inv_d);
        a_qs[i] = (int8_t)max(-128, min(127, v));
    }

    const uint8_t* w_qs_packed = w_ptr + 2;
    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int8_t tl[4], th[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t b = w_qs_packed[i*4 + j];
            tl[j] = (int8_t)(b & 0x0F);
            th[j] = (int8_t)((b >> 4) & 0x0F);
        }
        sumi = dp4a(*reinterpret_cast<int*>(tl), *reinterpret_cast<int*>(&a_qs[i*4]), sumi);
        sumi = dp4a(*reinterpret_cast<int*>(th), *reinterpret_cast<int*>(&a_qs[16+i*4]), sumi);
    }

    return d4_0 * (d8_1 * (float)sumi - 8.0f * asum);
}

// Small M: FP32 dequantization (memory-bound)
__global__ void gemm_small_fp32(
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

    // Unroll for ILP
    int kb = 0;
    for (; kb + 3 < num_blocks; kb += 4) {
        acc += q4_0_fp32_dot_fp32(weight_row_base + kb * 18, act_row + kb * QK);
        acc += q4_0_fp32_dot_fp32(weight_row_base + (kb + 1) * 18, act_row + (kb + 1) * QK);
        acc += q4_0_fp32_dot_fp32(weight_row_base + (kb + 2) * 18, act_row + (kb + 2) * QK);
        acc += q4_0_fp32_dot_fp32(weight_row_base + (kb + 3) * 18, act_row + (kb + 3) * QK);
    }
    for (; kb < num_blocks; kb++) {
        acc += q4_0_fp32_dot_fp32(weight_row_base + kb * 18, act_row + kb * QK);
    }

    output[m_idx * N + n_idx] = acc;
}

// Large M: DP4A with shared memory
__global__ void gemm_large_dp4a(
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

    __shared__ float s_act[QK * 4];

    const float* act_row = activation + m_idx * K;
    const uint8_t* weight_row_base = weight + n_base * num_blocks * 18;

    int kb = 0;
    const int chunk_size = 4;

    for (; kb + chunk_size <= num_blocks; kb += chunk_size) {
        int tidx = threadIdx.x;
        if (tidx < QK) {
            #pragma unroll
            for (int c = 0; c < chunk_size; c++) {
                s_act[c * QK + tidx] = act_row[(kb + c) * QK + tidx];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int c = 0; c < chunk_size; c++) {
            const float* a_ptr = s_act + c * QK;
            const uint8_t* w_ptr = weight_row_base + (kb + c) * 18;
            acc += q4_0_fp32_dot_dp4a(w_ptr, a_ptr);
        }

        __syncthreads();
    }

    for (; kb < num_blocks; kb++) {
        int tidx = threadIdx.x;
        if (tidx < QK) {
            s_act[tidx] = act_row[kb * QK + tidx];
        }
        __syncthreads();

        acc += q4_0_fp32_dot_dp4a(weight_row_base + kb * 18, s_act);

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

    // Hybrid dispatch
    if (M < 32) {
        gemm_small_fp32<<<grid, block>>>(
            (const uint8_t*)weight_contig.data_ptr<uint8_t>(),
            act_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        gemm_large_dp4a<<<grid, block>>>(
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
    m.def("forward", &forward, "W4A32C8 Q4_0 × FP32 Quantized GEMM (V5 - Hybrid FP32/DP4A)");
}
