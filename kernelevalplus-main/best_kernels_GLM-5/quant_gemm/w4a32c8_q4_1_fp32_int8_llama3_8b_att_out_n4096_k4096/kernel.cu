/**
 * Q4_1 Quantized GEMM for LLaMA-3-8B Attention Output - v12
 *
 * Optimization: Even smaller output tiles for maximum parallelism
 * - 16 outputs per block (256 blocks for N=4096)
 * - 2 outputs per warp (more K-blocks per thread)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int Q4_1_BLOCK = 20;
constexpr int NUM_K_BLOCKS = 4096 / 32;  // 128

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ void load_half2(const uint8_t* ptr, float& f0, float& f1) {
    uint32_t val = *reinterpret_cast<const uint32_t*>(ptr);
    f0 = __half2float(*reinterpret_cast<const half*>(&val));
    f1 = __half2float(reinterpret_cast<const half*>(&val)[1]);
}

struct ActBlock {
    float d_a;
    float asum;
    int packed[8];
};

__global__ void __launch_bounds__(256) gemm_kernel_small(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    __shared__ ActBlock act_shared[NUM_K_BLOCKS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int m = blockIdx.y;

    // Phase 1: Quantize activation
    const float* act_row = activation + static_cast<int64_t>(m) * K;

    for (int kb = tid; kb < NUM_K_BLOCKS; kb += 256) {
        const float* ap = act_row + kb * QK;

        float a[QK];
        float amax = 0.0f, asum = 0.0f;
        const float4* a4 = reinterpret_cast<const float4*>(ap);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = a4[i];
            a[i*4] = v.x; a[i*4+1] = v.y;
            a[i*4+2] = v.z; a[i*4+3] = v.w;
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
            asum += v.x + v.y + v.z + v.w;
        }

        float d_a = (amax > 1e-10f) ? (amax / 127.0f) : 1.0f;

        int packed[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            packed[i] = ((uint8_t)__float2int_rn(a[i*4]/d_a)) |
                       (((uint8_t)__float2int_rn(a[i*4+1]/d_a)) << 8) |
                       (((uint8_t)__float2int_rn(a[i*4+2]/d_a)) << 16) |
                       (((uint8_t)__float2int_rn(a[i*4+3]/d_a)) << 24);
            packed[i+4] = ((uint8_t)__float2int_rn(a[16+i*4]/d_a)) |
                         (((uint8_t)__float2int_rn(a[16+i*4+1]/d_a)) << 8) |
                         (((uint8_t)__float2int_rn(a[16+i*4+2]/d_a)) << 16) |
                         (((uint8_t)__float2int_rn(a[16+i*4+3]/d_a)) << 24);
        }

        act_shared[kb].d_a = d_a;
        act_shared[kb].asum = asum;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_shared[kb].packed[i] = packed[i];
        }
    }
    __syncthreads();

    // Phase 2: 16 outputs per block, 2 outputs per warp
    constexpr int outputs_per_block = 16;
    constexpr int outputs_per_warp = outputs_per_block / 8;  // 2

    const int n_start = blockIdx.x * outputs_per_block;
    const int warp_n_start = n_start + warp * outputs_per_warp;

    #pragma unroll
    for (int o = 0; o < outputs_per_warp; o++) {
        int n = warp_n_start + o;
        if (n >= N) continue;

        float sum = 0.0f;

        for (int kb = lane; kb < NUM_K_BLOCKS; kb += 32) {
            const uint8_t* wb = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;

            float d_w, m_w;
            load_half2(wb, d_w, m_w);

            const uint8_t* qs = wb + 4;
            int sumi = 0;

            const int* a_packed = act_shared[kb].packed;
            float d_a = act_shared[kb].d_a;
            float asum = act_shared[kb].asum;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint8_t b0 = qs[i*4], b1 = qs[i*4+1], b2 = qs[i*4+2], b3 = qs[i*4+3];
                int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
                int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) |
                          (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);

                sumi = dp4a(a_packed[i], w_lo, sumi);
                sumi = dp4a(a_packed[i+4], w_hi, sumi);
            }

            sum += d_w * d_a * (float)sumi + m_w * asum;
        }

        sum = warp_reduce(sum);

        if (lane == 0) {
            output[m * N + n] = sum;
        }
    }
}

__global__ void __launch_bounds__(256) gemm_kernel_large(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    float sum = 0.0f;
    const float* act = activation + static_cast<int64_t>(m) * K;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const uint8_t* wb = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
        const float* ap = act + kb * QK;

        float d_w, m_w;
        load_half2(wb, d_w, m_w);

        float a[QK];
        const float4* a4 = reinterpret_cast<const float4*>(ap);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = a4[i];
            a[i*4] = v.x; a[i*4+1] = v.y;
            a[i*4+2] = v.z; a[i*4+3] = v.w;
        }

        float amax = 0.0f, asum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            amax = fmaxf(amax, fabsf(a[i]));
            asum += a[i];
        }

        float d_a = (amax > 1e-10f) ? (amax / 127.0f) : 1.0f;

        const uint8_t* qs = wb + 4;
        int sumi = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t b0 = qs[i*4], b1 = qs[i*4+1], b2 = qs[i*4+2], b3 = qs[i*4+3];
            int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
            int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) |
                      (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);

            int a_lo = ((uint8_t)__float2int_rn(a[i*4]/d_a)) |
                      (((uint8_t)__float2int_rn(a[i*4+1]/d_a)) << 8) |
                      (((uint8_t)__float2int_rn(a[i*4+2]/d_a)) << 16) |
                      (((uint8_t)__float2int_rn(a[i*4+3]/d_a)) << 24);
            int a_hi = ((uint8_t)__float2int_rn(a[16+i*4]/d_a)) |
                      (((uint8_t)__float2int_rn(a[16+i*4+1]/d_a)) << 8) |
                      (((uint8_t)__float2int_rn(a[16+i*4+2]/d_a)) << 16) |
                      (((uint8_t)__float2int_rn(a[16+i*4+3]/d_a)) << 24);

            sumi = dp4a(a_lo, w_lo, sumi);
            sumi = dp4a(a_hi, w_hi, sumi);
        }

        sum += d_w * d_a * (float)sumi + m_w * asum;
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M < 16) {
        // N=4096, outputs_per_block=16 -> 256 blocks
        int num_blocks_n = (N + 15) / 16;
        dim3 block(256);
        dim3 grid(num_blocks_n, M);
        gemm_kernel_small<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_kernel_large<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 GEMM v12 for LLaMA-3-8B");
}
