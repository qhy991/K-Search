/**
 * W8A32C8 Q8_0 Quantized GEMM for DeepSeek-V3 Attention QKV Projection
 * Dimensions: M (variable) x N=1536 x K=7168
 *
 * v14: Combined strategy for optimal performance across all M
 * - M=1: Block-per-output with 64 threads (v11 pattern - best for M=1)
 * - M>1: Batched N outputs with 512 threads (v13 pattern - best for M>1)
 *
 * Performance targets:
 * - M=1: 1.39+ TFLOPS
 * - M=512: 2.0+ TFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;

typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size must be 34 bytes");

__device__ __forceinline__ float read_half(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h; return __half2float(un.f16);
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int r; asm volatile("dp4a.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
#else
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
}
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffff, val, o);
    return val;
}

//=============================================================================
// Kernel 1: Block-per-output with 64 threads (best for M=1)
//=============================================================================
constexpr int BLOCK_DIM_64 = 64;

__launch_bounds__(BLOCK_DIM_64)
__global__ void gemm_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int n = blockIdx.x;

    if (m >= M || n >= N) return;

    const int num_blocks_k = K / QK;
    float sum = 0.0f;

    for (int block_k = tid; block_k < num_blocks_k; block_k += BLOCK_DIM_64) {
        const int k_start = block_k * QK;

        const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
            weight + (n * num_blocks_k + block_k) * sizeof(block_q8_0)
        );
        const float d_w = read_half(wb->d);

        const float4* a_vec = reinterpret_cast<const float4*>(activation + m * K + k_start);

        float a_block[32];
        float a_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = a_vec[i];
            a_block[i*4] = v.x;
            a_block[i*4+1] = v.y;
            a_block[i*4+2] = v.z;
            a_block[i*4+3] = v.w;
            a_max = fmaxf(a_max, fabsf(v.x));
            a_max = fmaxf(a_max, fabsf(v.y));
            a_max = fmaxf(a_max, fabsf(v.z));
            a_max = fmaxf(a_max, fabsf(v.w));
        }

        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4] / d_a);
            int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);

            int a_pack = (int((uint8_t)q0)) | (int((uint8_t)q1) << 8) |
                         (int((uint8_t)q2) << 16) | (int((uint8_t)q3) << 24);

            int w_pack = (int((uint8_t)wb->qs[i * 4])) |
                         (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                         (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                         (int((uint8_t)wb->qs[i * 4 + 3]) << 24);

            sumi = dp4a(a_pack, w_pack, sumi);
        }

        sum += d_w * d_a * (float)sumi;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float s_data[2];
    int wid = tid >> 5;
    int lane = tid & 31;

    if (lane == 0) s_data[wid] = sum;
    __syncthreads();

    if (tid == 0) output[m * N + n] = s_data[0] + s_data[1];
}

//=============================================================================
// Kernel 2: Batched N outputs (best for M>1)
//=============================================================================
constexpr int TILE_N = 16;
constexpr int BLOCK_DIM_512 = 512;

__launch_bounds__(BLOCK_DIM_512)
__global__ void gemm_batched_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int m = blockIdx.y;
    const int n_base = blockIdx.x * TILE_N;

    if (m >= M) return;

    const int n = n_base + warp_id;
    const bool valid_n = n < N;

    const int num_blocks_k = K / QK;
    __shared__ float s_sums[TILE_N];

    float sum = 0.0f;

    for (int block_k = lane_id; block_k < num_blocks_k; block_k += 32) {
        const int k_start = block_k * QK;

        const float4* a_vec = reinterpret_cast<const float4*>(activation + m * K + k_start);

        float a_block[32];
        float a_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = a_vec[i];
            a_block[i*4] = v.x;
            a_block[i*4+1] = v.y;
            a_block[i*4+2] = v.z;
            a_block[i*4+3] = v.w;
            a_max = fmaxf(a_max, fabsf(v.x));
            a_max = fmaxf(a_max, fabsf(v.y));
            a_max = fmaxf(a_max, fabsf(v.z));
            a_max = fmaxf(a_max, fabsf(v.w));
        }

        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        if (valid_n) {
            const block_q8_0* wb = reinterpret_cast<const block_q8_0*>(
                weight + (n * num_blocks_k + block_k) * sizeof(block_q8_0)
            );
            const float d_w = read_half(wb->d);

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);

                int a_pack = (int((uint8_t)q0)) | (int((uint8_t)q1) << 8) |
                             (int((uint8_t)q2) << 16) | (int((uint8_t)q3) << 24);

                int w_pack = (int((uint8_t)wb->qs[i * 4])) |
                             (int((uint8_t)wb->qs[i * 4 + 1]) << 8) |
                             (int((uint8_t)wb->qs[i * 4 + 2]) << 16) |
                             (int((uint8_t)wb->qs[i * 4 + 3]) << 24);

                sumi = dp4a(a_pack, w_pack, sumi);
            }

            sum += d_w * d_a * (float)sumi;
        }
    }

    sum = warp_reduce_sum(sum);

    if (lane_id == 0 && valid_n) s_sums[warp_id] = sum;
    __syncthreads();

    if (tid < TILE_N && (n_base + tid) < N) {
        output[m * N + n_base + tid] = s_sums[tid];
    }
}

//=============================================================================
// Host interface with strategy dispatch
//=============================================================================
torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kUInt8);
    TORCH_CHECK(activation.is_cuda() && activation.dtype() == torch::kFloat32);

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        // Use block-per-output with 64 threads for M=1
        dim3 grid(N, M);
        gemm_m1_kernel<<<grid, BLOCK_DIM_64>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        // Use batched kernel for M>1
        const int blocks_x = (N + TILE_N - 1) / TILE_N;
        dim3 grid(blocks_x, M);
        gemm_batched_kernel<<<grid, BLOCK_DIM_512>>>(
            weight.data_ptr<uint8_t>(), activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W8A32C8 Q8_0 Quantized GEMM v14");
}
