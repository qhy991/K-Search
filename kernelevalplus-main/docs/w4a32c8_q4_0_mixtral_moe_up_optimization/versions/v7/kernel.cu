/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel - V7
 * 
 * Based on V2 which achieved best performance for small batches.
 * Added optimizations for larger batches.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Process a single Q4_0 block - fully unrolled for best ILP
#define PROCESS_BLOCK(wb, ap, sc) ({ \
    float _b0 = (float)((int)((wb)->qs[0] & 0x0F) - 8) * (ap)[0] + (float)((int)((wb)->qs[0] >> 4) - 8) * (ap)[16]; \
    float _b1 = (float)((int)((wb)->qs[1] & 0x0F) - 8) * (ap)[1] + (float)((int)((wb)->qs[1] >> 4) - 8) * (ap)[17]; \
    float _b2 = (float)((int)((wb)->qs[2] & 0x0F) - 8) * (ap)[2] + (float)((int)((wb)->qs[2] >> 4) - 8) * (ap)[18]; \
    float _b3 = (float)((int)((wb)->qs[3] & 0x0F) - 8) * (ap)[3] + (float)((int)((wb)->qs[3] >> 4) - 8) * (ap)[19]; \
    float _b4 = (float)((int)((wb)->qs[4] & 0x0F) - 8) * (ap)[4] + (float)((int)((wb)->qs[4] >> 4) - 8) * (ap)[20]; \
    float _b5 = (float)((int)((wb)->qs[5] & 0x0F) - 8) * (ap)[5] + (float)((int)((wb)->qs[5] >> 4) - 8) * (ap)[21]; \
    float _b6 = (float)((int)((wb)->qs[6] & 0x0F) - 8) * (ap)[6] + (float)((int)((wb)->qs[6] >> 4) - 8) * (ap)[22]; \
    float _b7 = (float)((int)((wb)->qs[7] & 0x0F) - 8) * (ap)[7] + (float)((int)((wb)->qs[7] >> 4) - 8) * (ap)[23]; \
    float _b8 = (float)((int)((wb)->qs[8] & 0x0F) - 8) * (ap)[8] + (float)((int)((wb)->qs[8] >> 4) - 8) * (ap)[24]; \
    float _b9 = (float)((int)((wb)->qs[9] & 0x0F) - 8) * (ap)[9] + (float)((int)((wb)->qs[9] >> 4) - 8) * (ap)[25]; \
    float _b10 = (float)((int)((wb)->qs[10] & 0x0F) - 8) * (ap)[10] + (float)((int)((wb)->qs[10] >> 4) - 8) * (ap)[26]; \
    float _b11 = (float)((int)((wb)->qs[11] & 0x0F) - 8) * (ap)[11] + (float)((int)((wb)->qs[11] >> 4) - 8) * (ap)[27]; \
    float _b12 = (float)((int)((wb)->qs[12] & 0x0F) - 8) * (ap)[12] + (float)((int)((wb)->qs[12] >> 4) - 8) * (ap)[28]; \
    float _b13 = (float)((int)((wb)->qs[13] & 0x0F) - 8) * (ap)[13] + (float)((int)((wb)->qs[13] >> 4) - 8) * (ap)[29]; \
    float _b14 = (float)((int)((wb)->qs[14] & 0x0F) - 8) * (ap)[14] + (float)((int)((wb)->qs[14] >> 4) - 8) * (ap)[30]; \
    float _b15 = (float)((int)((wb)->qs[15] & 0x0F) - 8) * (ap)[15] + (float)((int)((wb)->qs[15] >> 4) - 8) * (ap)[31]; \
    (sc) * (_b0 + _b1 + _b2 + _b3 + _b4 + _b5 + _b6 + _b7 + _b8 + _b9 + _b10 + _b11 + _b12 + _b13 + _b14 + _b15); \
})

// ============================================================================
// Small batch kernel (M <= 16): Shared memory + fully unrolled
// ============================================================================
__global__ void __launch_bounds__(128) gemm_small_batch(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n_base = blockIdx.x * 128;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (n_base >= N || m >= M) return;
    
    __shared__ float act_shared[4096];
    
    const float4* act_vec = reinterpret_cast<const float4*>(activation + m * K);
    float4* act_shared_vec = reinterpret_cast<float4*>(act_shared);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        if (idx < K / 4) act_shared_vec[idx] = act_vec[idx];
    }
    __syncthreads();
    
    const int n = n_base + tid;
    if (n >= N) return;
    
    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    float sum = 0.0f;
    
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0* wb = &w_row[kb];
        const float sc = half_to_float(wb->d);
        const float* ap = act_shared + kb * 32;
        sum += PROCESS_BLOCK(wb, ap, sc);
    }
    
    output[m * N + n] = sum;
}

// ============================================================================
// Large batch kernel (M > 16)
// ============================================================================
__global__ void __launch_bounds__(256) gemm_large_batch(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    
    if (n >= N || m >= M) return;
    
    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    const float* act_row = activation + m * K;
    float sum = 0.0f;
    
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0* wb = &w_row[kb];
        const float sc = half_to_float(wb->d);
        const float* ap = act_row + kb * 32;
        sum += PROCESS_BLOCK(wb, ap, sc);
    }
    
    output[m * N + n] = sum;
}

// ============================================================================
// PyTorch binding
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));
    
    if (M <= 16) {
        dim3 block(128);
        dim3 grid((N + 127) / 128, M);
        gemm_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_large_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
