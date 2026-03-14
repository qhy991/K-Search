/**
 * Optimized Q4_0 GEMM for DeepSeek-V2 LM Head - v10
 * Corrected activation indexing
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int WARP_SIZE = 32;
constexpr int Q4_0_BLOCK = 18;
constexpr int NUM_K_BLOCKS = 160;

__device__ __forceinline__ float half_to_float(const uint8_t* ptr) {
    uint16_t h = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

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

__global__ void __launch_bounds__(128) gemm_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g_warp = (blockIdx.x << 2) + warp;
    const int n_warps = gridDim.x << 2;

    for (int base = g_warp << 2; base < M * N; base += n_warps << 2) {
        float p0 = 0, p1 = 0, p2 = 0, p3 = 0;

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            int idx = base + o;
            if (idx >= M * N) continue;

            int m = idx / N;
            int n = idx % N;
            float sum = 0;

            for (int kb = lane; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
                const uint8_t* wb = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
                const float* ap = activation + m * K + kb * QK;

                float dw = half_to_float(wb);

                // Load all 32 activation values
                float a[QK];
                const float4* a4 = reinterpret_cast<const float4*>(ap);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float4 v = a4[i];
                    a[i*4] = v.x; a[i*4+1] = v.y; a[i*4+2] = v.z; a[i*4+3] = v.w;
                }

                float amax = 0, asum = 0;
                #pragma unroll
                for (int i = 0; i < QK; i++) {
                    amax = fmaxf(amax, fabsf(a[i]));
                    asum += a[i];
                }

                float da = (amax > 1e-10f) ? (amax / 127.0f) : 1.0f;
                float sc = 127.0f / fmaxf(amax, 1e-10f);

                const uint8_t* qs = wb + 2;
                int sumi = 0;

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    uint8_t b0 = qs[i*4], b1 = qs[i*4+1], b2 = qs[i*4+2], b3 = qs[i*4+3];

                    int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
                    int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) | (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);

                    int a_lo = ((uint8_t)__float2int_rn(a[i*4]*sc)) | (((uint8_t)__float2int_rn(a[i*4+1]*sc)) << 8) |
                              (((uint8_t)__float2int_rn(a[i*4+2]*sc)) << 16) | (((uint8_t)__float2int_rn(a[i*4+3]*sc)) << 24);
                    int a_hi = ((uint8_t)__float2int_rn(a[16+i*4]*sc)) | (((uint8_t)__float2int_rn(a[16+i*4+1]*sc)) << 8) |
                              (((uint8_t)__float2int_rn(a[16+i*4+2]*sc)) << 16) | (((uint8_t)__float2int_rn(a[16+i*4+3]*sc)) << 24);

                    sumi = dp4a(a_lo, w_lo, sumi);
                    sumi = dp4a(a_hi, w_hi, sumi);
                }

                sum += dw * (da * (float)sumi - 8.0f * asum);
            }

            if (o == 0) p0 = sum;
            else if (o == 1) p1 = sum;
            else if (o == 2) p2 = sum;
            else p3 = sum;
        }

        p0 = warp_reduce(p0);
        p1 = warp_reduce(p1);
        p2 = warp_reduce(p2);
        p3 = warp_reduce(p3);

        if (lane == 0) {
            if (base < M * N) output[base] = p0;
            if (base + 1 < M * N) output[base + 1] = p1;
            if (base + 2 < M * N) output[base + 2] = p2;
            if (base + 3 < M * N) output[base + 3] = p3;
        }
    }
}

__global__ void __launch_bounds__(256) gemm_large_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    float sum = 0;
    const float* act = activation + m * K;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const uint8_t* wb = weight + (static_cast<int64_t>(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
        const float* ap = act + kb * QK;

        float dw = half_to_float(wb);

        float a[QK];
        const float4* a4 = reinterpret_cast<const float4*>(ap);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = a4[i];
            a[i*4] = v.x; a[i*4+1] = v.y; a[i*4+2] = v.z; a[i*4+3] = v.w;
        }

        float amax = 0, asum = 0;
        #pragma unroll
        for (int i = 0; i < QK; i++) {
            amax = fmaxf(amax, fabsf(a[i]));
            asum += a[i];
        }

        float da = (amax > 1e-10f) ? (amax / 127.0f) : 1.0f;
        float sc = 127.0f / fmaxf(amax, 1e-10f);

        const uint8_t* qs = wb + 2;
        int sumi = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t b0 = qs[i*4], b1 = qs[i*4+1], b2 = qs[i*4+2], b3 = qs[i*4+3];

            int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
            int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) | (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);

            int a_lo = ((uint8_t)__float2int_rn(a[i*4]*sc)) | (((uint8_t)__float2int_rn(a[i*4+1]*sc)) << 8) |
                      (((uint8_t)__float2int_rn(a[i*4+2]*sc)) << 16) | (((uint8_t)__float2int_rn(a[i*4+3]*sc)) << 24);
            int a_hi = ((uint8_t)__float2int_rn(a[16+i*4]*sc)) | (((uint8_t)__float2int_rn(a[16+i*4+1]*sc)) << 8) |
                      (((uint8_t)__float2int_rn(a[16+i*4+2]*sc)) << 16) | (((uint8_t)__float2int_rn(a[16+i*4+3]*sc)) << 24);

            sumi = dp4a(a_lo, w_lo, sumi);
            sumi = dp4a(a_hi, w_hi, sumi);
        }

        sum += dw * (da * (float)sumi - 8.0f * asum);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 64) {
        gemm_kernel<<<1024, 128>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_large_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM v10");
}
