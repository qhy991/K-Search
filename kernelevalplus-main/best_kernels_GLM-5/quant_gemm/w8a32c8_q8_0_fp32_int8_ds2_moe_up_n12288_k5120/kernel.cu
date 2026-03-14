/**
 * W8A32C8: Q8_0 weight x FP32 activation GEMM kernel - Optimized v1
 *
 * DeepSeek-V2 MoE Up/Gate projection: N=12288, K=5120
 *
 * Key optimizations:
 * 1. M=1: Warp-based approach with 1024 threads per block
 * 2. Shared memory for activation quantization cache
 * 3. DP4A instruction for INT8 dot products
 * 4. Multiple kernel variants for different batch sizes
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

#define QK 32
#define WARP_SIZE 32

struct __align__(2) block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

// ============================================================================
// M=1 Kernel: Warp-based with 1024 threads (best performer)
// ============================================================================
__global__ void __launch_bounds__(1024) gemm_q8_0_m1_warp(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K)
{
    extern __shared__ char smem_raw[];
    float* s_scale_a = reinterpret_cast<float*>(smem_raw);
    int8_t* s_act_qs = reinterpret_cast<int8_t*>(s_scale_a + K/QK);

    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Phase 1: Cooperatively quantize and cache activation
    for (int kb = tid; kb < num_blocks_k; kb += blockDim.x) {
        const float* act_ptr = activation + kb * QK;

        float a[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 val = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a[i * 4 + 0] = val.x;
            a[i * 4 + 1] = val.y;
            a[i * 4 + 2] = val.z;
            a[i * 4 + 3] = val.w;
        }

        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            amax = fmaxf(amax, fabsf(a[i]));
        }
        const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

        s_scale_a[kb] = scale_a;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            s_act_qs[kb * 32 + i] = (int8_t)__float2int_rn(a[i] * inv_scale_a);
        }
    }

    __syncthreads();

    // Phase 2: Compute outputs
    const int n = blockIdx.x * num_warps + warp;
    if (n >= N) return;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    float sum = 0.0f;

    for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
        const block_q8_0 w_block = w_blocks[n * num_blocks_k + kb];
        const float scale_w = read_half_as_float(w_block.d);
        const float scale_a = s_scale_a[kb];
        const int8_t* act_qs = &s_act_qs[kb * 32];

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 4;
            int w_packed = *reinterpret_cast<const int*>(&w_block.qs[idx]);
            int a_packed = *reinterpret_cast<const int*>(&act_qs[idx]);
            sumi = dp4a(w_packed, a_packed, sumi);
        }

        sum += scale_w * scale_a * (float)sumi;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[n] = sum;
    }
}

// ============================================================================
// Small batch kernel with warp-based parallelism
// ============================================================================
__global__ void __launch_bounds__(512) gemm_q8_0_small_batch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int idx = blockIdx.x * num_warps + warp;
    if (idx >= M * N) return;

    const int m = idx / N;
    const int n = idx % N;

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const float* act_row = activation + (long long)m * K;

    float sum = 0.0f;

    for (int kb = lane; kb < num_blocks_k; kb += WARP_SIZE) {
        const block_q8_0 w_block = w_blocks[n * num_blocks_k + kb];
        const float scale_w = read_half_as_float(w_block.d);

        const float* act_ptr = act_row + kb * QK;

        float4 act_vec[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_vec[i] = *reinterpret_cast<const float4*>(act_ptr + i * 4);
        }

        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            amax = fmaxf(amax, fabsf(act_vec[i].x));
            amax = fmaxf(amax, fabsf(act_vec[i].y));
            amax = fmaxf(amax, fabsf(act_vec[i].z));
            amax = fmaxf(amax, fabsf(act_vec[i].w));
        }

        const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

        int8_t act_qs[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            act_qs[i * 4 + 0] = (int8_t)__float2int_rn(act_vec[i].x * inv_scale_a);
            act_qs[i * 4 + 1] = (int8_t)__float2int_rn(act_vec[i].y * inv_scale_a);
            act_qs[i * 4 + 2] = (int8_t)__float2int_rn(act_vec[i].z * inv_scale_a);
            act_qs[i * 4 + 3] = (int8_t)__float2int_rn(act_vec[i].w * inv_scale_a);
        }

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = i * 4;
            int w_packed = *reinterpret_cast<const int*>(&w_block.qs[idx]);
            int a_packed = *reinterpret_cast<int*>(&act_qs[idx]);
            sumi = dp4a(w_packed, a_packed, sumi);
        }

        sum += scale_w * scale_a * (float)sumi;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = sum;
    }
}

// ============================================================================
// Large batch kernel with K-batch processing
// ============================================================================
constexpr int TILE_M = 16;
constexpr int TILE_N = 32;
constexpr int K_BATCH = 8;

__global__ void __launch_bounds__(512) gemm_q8_0_large_batch_kbatch(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int m = block_m + warp;
    const int n = block_n + lane;

    const bool valid = (m < M) && (n < N);

    const block_q8_0* w_blocks = reinterpret_cast<const block_q8_0*>(weight);
    const int num_blocks_k = K / QK;
    const int num_batches = (num_blocks_k + K_BATCH - 1) / K_BATCH;

    __shared__ float s_scales[K_BATCH][TILE_N];
    __shared__ int8_t s_qs[K_BATCH][TILE_N][32];

    float sum = 0.0f;

    for (int batch = 0; batch < num_batches; batch++) {
        const int kb_start = batch * K_BATCH;
        const int kb_end = min(kb_start + K_BATCH, num_blocks_k);
        const int actual_batch = kb_end - kb_start;

        for (int k = 0; k < actual_batch; k++) {
            const int kb = kb_start + k;
            const int load_n = block_n + lane;

            if (load_n < N && warp == 0) {
                const block_q8_0 w_block = w_blocks[load_n * num_blocks_k + kb];
                s_scales[k][lane] = read_half_as_float(w_block.d);
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    s_qs[k][lane][i] = w_block.qs[i];
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < actual_batch; k++) {
            const int kb = kb_start + k;

            if (valid) {
                const float* act_ptr = activation + (long long)m * K + kb * QK;

                float4 act_vec[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    act_vec[i] = *reinterpret_cast<const float4*>(act_ptr + i * 4);
                }

                float amax = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    amax = fmaxf(amax, fabsf(act_vec[i].x));
                    amax = fmaxf(amax, fabsf(act_vec[i].y));
                    amax = fmaxf(amax, fabsf(act_vec[i].z));
                    amax = fmaxf(amax, fabsf(act_vec[i].w));
                }

                const float scale_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                const float inv_scale_a = (amax > 0.0f) ? (127.0f / amax) : 1.0f;

                int8_t act_qs[32];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    act_qs[i * 4 + 0] = (int8_t)__float2int_rn(act_vec[i].x * inv_scale_a);
                    act_qs[i * 4 + 1] = (int8_t)__float2int_rn(act_vec[i].y * inv_scale_a);
                    act_qs[i * 4 + 2] = (int8_t)__float2int_rn(act_vec[i].z * inv_scale_a);
                    act_qs[i * 4 + 3] = (int8_t)__float2int_rn(act_vec[i].w * inv_scale_a);
                }

                const float scale_w = s_scales[k][lane];
                int sumi = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int idx = i * 4;
                    int w_packed = *reinterpret_cast<int*>(&s_qs[k][lane][idx]);
                    int a_packed = *reinterpret_cast<int*>(&act_qs[idx]);
                    sumi = dp4a(w_packed, a_packed, sumi);
                }

                sum += scale_w * scale_a * (float)sumi;
            }
        }

        __syncthreads();
    }

    if (valid) {
        output[(long long)m * N + n] = sum;
    }
}

// ============================================================================
// PyTorch binding
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K)
{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(K % QK == 0, "K must be multiple of 32");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));
    size_t smem_size = (K / QK) * sizeof(float) + K * sizeof(int8_t);

    if (M == 1) {
        // Use warp-based approach (best performer)
        const int threads = 1024;
        const int num_warps = threads / WARP_SIZE;
        const int blocks = (N + num_warps - 1) / num_warps;

        gemm_q8_0_m1_warp<<<blocks, threads, smem_size>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );
    }
    else if (M <= 16) {
        const int threads = 512;
        const int num_warps = threads / WARP_SIZE;
        const int blocks = (M * N + num_warps - 1) / num_warps;

        gemm_q8_0_small_batch<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
    else {
        dim3 block(512);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        gemm_q8_0_large_batch_kbatch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q8_0 x FP32_INT8 GEMM for DeepSeek-V2 MoE Up/Gate");
}
