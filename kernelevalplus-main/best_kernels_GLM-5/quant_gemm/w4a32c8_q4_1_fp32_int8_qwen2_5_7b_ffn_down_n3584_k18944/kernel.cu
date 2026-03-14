/**
 * Optimized Quantized GEMM Kernel - Version 13 (Combined)
 * 
 * Strategy dispatch based on batch size M:
 * - M=1: Cooperative kernel (activation broadcast via shared memory)
 * - M=2-8: Simple kernel (one output per thread)
 * - M>=64: 4x kernel (4 outputs per thread)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

constexpr int QK = 32;
constexpr int Q4_1_BLOCK = 20;
constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

// ============================================================================
// Kernel 1: Cooperative (best for M=1)
// ============================================================================
constexpr int OUTPUTS_PER_BLOCK = 8;

__global__ void __launch_bounds__(BLOCK_SIZE) gemm_kernel_cooperative(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int block_n = blockIdx.x * OUTPUTS_PER_BLOCK;
    
    if (m >= M) return;

    const int num_k_blocks = K / QK;

    int n_cols[OUTPUTS_PER_BLOCK];
    bool valid[OUTPUTS_PER_BLOCK];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_BLOCK; i++) {
        n_cols[i] = block_n + i;
        valid[i] = (n_cols[i] < N);
    }

    __shared__ int8_t sh_a_qs[QK];
    __shared__ float sh_d_a;
    __shared__ float sh_act_sum;
    
    float sums[OUTPUTS_PER_BLOCK];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_BLOCK; i++) sums[i] = 0.0f;

    const int col_idx = tid / 32;
    const int lane = tid % 32;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_base = kb * QK;

        if (tid < 32) {
            const float* act_ptr = activation + m * K + k_base;
            float val = act_ptr[tid];
            float act_max = fabsf(val);
            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                act_max = fmaxf(act_max, __shfl_down_sync(0xffffffff, act_max, offset));
            }
            
            if (tid == 0) {
                sh_d_a = (act_max > 0.0f) ? (act_max / 127.0f) : 1.0f;
            }
            __syncwarp();
            
            int8_t q = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(val / sh_d_a))));
            sh_a_qs[tid] = q;
            
            float act_sum = val;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                act_sum += __shfl_down_sync(0xffffffff, act_sum, offset);
            }
            if (tid == 0) {
                sh_act_sum = act_sum;
            }
        }

        __syncthreads();

        if (col_idx < OUTPUTS_PER_BLOCK && valid[col_idx]) {
            const int n = n_cols[col_idx];
            const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_1_BLOCK;
            
            float d_w, m_w;
            if (lane == 0) {
                d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
                m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
            }
            d_w = __shfl_sync(0xffffffff, d_w, 0);
            m_w = __shfl_sync(0xffffffff, m_w, 0);
            
            int partial_sumi = 0;
            if (lane < 16) {
                uint8_t packed = w_block[4 + lane];
                partial_sumi = (packed & 0x0F) * sh_a_qs[lane] + ((packed >> 4) & 0x0F) * sh_a_qs[lane + 16];
            }
            
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                if (lane < 16) {
                    partial_sumi += __shfl_down_sync(0xffff, partial_sumi, offset);
                }
            }
            
            if (lane == 0) {
                sums[col_idx] += d_w * sh_d_a * static_cast<float>(partial_sumi) + m_w * sh_act_sum;
            }
        }

        __syncthreads();
    }

    if (lane == 0 && col_idx < OUTPUTS_PER_BLOCK && valid[col_idx]) {
        output[m * N + n_cols[col_idx]] = sums[col_idx];
    }
}

// ============================================================================
// Kernel 2: Simple (best for M=2-8)
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE) gemm_kernel_simple(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    const int num_k_blocks = K / QK;
    const float* act_row = activation + m * K;

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_base = kb * QK;
        const float4* act_ptr = reinterpret_cast<const float4*>(act_row + k_base);
        
        float act_max = 0.0f;
        float act_sum = 0.0f;
        float4 v[8];
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            v[i] = act_ptr[i];
            act_max = fmaxf(act_max, fabsf(v[i].x));
            act_max = fmaxf(act_max, fabsf(v[i].y));
            act_max = fmaxf(act_max, fabsf(v[i].z));
            act_max = fmaxf(act_max, fabsf(v[i].w));
            act_sum += v[i].x + v[i].y + v[i].z + v[i].w;
        }

        const float d_a = (act_max > 0.0f) ? (act_max / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        int8_t a_qs[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int base = i * 4;
            a_qs[base + 0] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].x * inv_d_a))));
            a_qs[base + 1] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].y * inv_d_a))));
            a_qs[base + 2] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].z * inv_d_a))));
            a_qs[base + 3] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].w * inv_d_a))));
        }

        const uint8_t* w_block = weight + (n * num_k_blocks + kb) * Q4_1_BLOCK;
        const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
        const float m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
        const uint8_t* w_qs = w_block + 4;

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t p = w_qs[i];
            sumi += (p & 0x0F) * a_qs[i];
            sumi += ((p >> 4) & 0x0F) * a_qs[i + 16];
        }

        sum += d_w * d_a * static_cast<float>(sumi) + m_w * act_sum;
    }

    output[m * N + n] = sum;
}

// ============================================================================
// Kernel 3: 4x (best for M>=64)
// ============================================================================
__global__ void __launch_bounds__(64) gemm_kernel_4x(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int n_base = blockIdx.x * 64 * 4 + tid * 4;

    if (m >= M) return;

    const int num_k_blocks = K / QK;
    const float* act_row = activation + m * K;

    int n[4] = {n_base, n_base + 1, n_base + 2, n_base + 3};
    bool valid[4] = {n[0] < N, n[1] < N, n[2] < N, n[3] < N};
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_base = kb * QK;
        const float4* act_ptr = reinterpret_cast<const float4*>(act_row + k_base);
        
        float act_max = 0.0f;
        float act_sum = 0.0f;
        float4 v[8];
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            v[i] = act_ptr[i];
            act_max = fmaxf(act_max, fabsf(v[i].x));
            act_max = fmaxf(act_max, fabsf(v[i].y));
            act_max = fmaxf(act_max, fabsf(v[i].z));
            act_max = fmaxf(act_max, fabsf(v[i].w));
            act_sum += v[i].x + v[i].y + v[i].z + v[i].w;
        }

        const float d_a = (act_max > 0.0f) ? (act_max / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        int8_t a_qs[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int base = i * 4;
            a_qs[base + 0] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].x * inv_d_a))));
            a_qs[base + 1] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].y * inv_d_a))));
            a_qs[base + 2] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].z * inv_d_a))));
            a_qs[base + 3] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(v[i].w * inv_d_a))));
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (!valid[j]) continue;

            const uint8_t* w_block = weight + (n[j] * num_k_blocks + kb) * Q4_1_BLOCK;
            const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
            const float m_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
            const uint8_t* w_qs = w_block + 4;

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t p = w_qs[i];
                sumi += (p & 0x0F) * a_qs[i];
                sumi += ((p >> 4) & 0x0F) * a_qs[i + 16];
            }

            sums[j] += d_w * d_a * static_cast<float>(sumi) + m_w * act_sum;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (valid[j]) {
            output[m * N + n[j]] = sums[j];
        }
    }
}

// ============================================================================
// Forward function with strategy dispatch
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        // M=1: Cooperative kernel (activation broadcast)
        dim3 grid((N + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK, M);
        dim3 block(BLOCK_SIZE);
        gemm_kernel_cooperative<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M < 64) {
        // M=2-8: Simple kernel (one output per thread)
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
        dim3 block(BLOCK_SIZE);
        gemm_kernel_simple<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // M>=64: 4x kernel (4 outputs per thread)
        dim3 grid((N + 255) / 256, M);
        dim3 block(64);
        gemm_kernel_4x<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM Q4_1 v13 Combined");
}
