/**
 * Quantized GEMM for DeepSeek-V3 MoE Routing Down Projection - v19
 *
 * Combined kernel with optimal dispatch for all M ranges:
 * - M=1: 128 blocks × 128 threads
 * - M=2: 128 blocks × 256 threads
 * - M=3-4: 256 blocks × 256 threads
 * - M=5-8: 512 blocks × 256 threads
 * - M>=16: Shared memory kernel for better weight reuse
 *
 * Performance target:
 * - M=1: ~470 GFLOPS
 * - M=512: ~2000 GFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int Q4_0_BLOCK = 18;
constexpr int NUM_K_BLOCKS = 224;

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
           a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float compute_block(
    const uint8_t* w_block,
    const float* act_ptr
) {
    float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
    const uint8_t* qs = w_block + 2;

    float a_vals[QK];
    float a_max = 0.0f;
    float a_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < QK; i += 4) {
        float4 v = *reinterpret_cast<const float4*>(act_ptr + i);
        a_vals[i] = v.x; a_vals[i+1] = v.y;
        a_vals[i+2] = v.z; a_vals[i+3] = v.w;
    }

    #pragma unroll
    for (int i = 0; i < QK; i++) {
        a_max = fmaxf(a_max, fabsf(a_vals[i]));
        a_sum += a_vals[i];
    }

    float d_a = fmaxf(a_max / 127.0f, 1e-10f);
    float scale = 127.0f / fmaxf(a_max, 1e-10f);

    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t b0 = qs[i*4+0], b1 = qs[i*4+1], b2 = qs[i*4+2], b3 = qs[i*4+3];

        int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
        int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) |
                   (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);

        int a_lo = (int)(uint8_t)lroundf(a_vals[i*4] * scale) |
                   ((int)(uint8_t)lroundf(a_vals[i*4+1] * scale) << 8) |
                   ((int)(uint8_t)lroundf(a_vals[i*4+2] * scale) << 16) |
                   ((int)(uint8_t)lroundf(a_vals[i*4+3] * scale) << 24);

        int a_hi = (int)(uint8_t)lroundf(a_vals[16+i*4] * scale) |
                   ((int)(uint8_t)lroundf(a_vals[16+i*4+1] * scale) << 8) |
                   ((int)(uint8_t)lroundf(a_vals[16+i*4+2] * scale) << 16) |
                   ((int)(uint8_t)lroundf(a_vals[16+i*4+3] * scale) << 24);

        sumi = dp4a(a_lo, w_lo, sumi);
        sumi = dp4a(a_hi, w_hi, sumi);
    }

    return d_w * (d_a * sumi - 8.0f * a_sum);
}

// Warp-based kernel for small M
__global__ void __launch_bounds__(256) gemm_warp_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    const int total_warps = gridDim.x * warps_per_block;

    constexpr int OUTPUTS_PER_WARP = 4;

    for (int base = global_warp_id * OUTPUTS_PER_WARP; base < M * N; base += total_warps * OUTPUTS_PER_WARP) {
        int m_idx[OUTPUTS_PER_WARP], n_idx[OUTPUTS_PER_WARP];
        float sums[OUTPUTS_PER_WARP] = {0.0f};

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            int idx = base + o;
            m_idx[o] = (idx < M * N) ? idx / N : -1;
            n_idx[o] = (idx < M * N) ? idx % N : 0;
        }

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += 32) {
            int k_start = kb * QK;

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (m_idx[o] < 0) continue;

                const uint8_t* w_block = weight + (int64_t(n_idx[o]) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
                const float* a_ptr = activation + int64_t(m_idx[o]) * K + k_start;

                sums[o] += compute_block(w_block, a_ptr);
            }
        }

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            sums[o] = warp_reduce_sum(sums[o]);
            if (lane_id == 0 && m_idx[o] >= 0) {
                output[int64_t(m_idx[o]) * N + n_idx[o]] = sums[o];
            }
        }
    }
}

// Shared memory kernel for large M
__global__ void __launch_bounds__(256) gemm_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    extern __shared__ char shared_mem[];
    float* s_act_scales = reinterpret_cast<float*>(shared_mem);
    float* s_act_sums = reinterpret_cast<float*>(shared_mem + NUM_K_BLOCKS * sizeof(float));
    int8_t* s_act_qs = reinterpret_cast<int8_t*>(shared_mem + NUM_K_BLOCKS * 2 * sizeof(float));

    const float* act_row = activation + int64_t(m) * K;

    // Cooperatively load and quantize activation
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_base = kb * QK;

        float act_max = 0.0f;
        float act_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            const float val = __ldg(&act_row[k_base + i]);
            act_max = fmaxf(act_max, fabsf(val));
            act_sum += val;
        }

        const float scale = fmaxf(act_max / 127.0f, 1e-10f);
        s_act_scales[kb] = scale;
        s_act_sums[kb] = act_sum;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            const float val = __ldg(&act_row[k_base + i]);
            const float q = roundf(val / scale);
            s_act_qs[k_base + i] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, q)));
        }
    }
    __syncthreads();

    float sum = 0.0f;

    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        const float act_scale = s_act_scales[kb];
        const float act_sum = s_act_sums[kb];
        const int8_t* act_qs = &s_act_qs[kb * QK];

        const uint8_t* w_block = weight + (int64_t(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
        float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* qs = w_block + 2;

        int int_sum = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int a_packed = *reinterpret_cast<const int*>(&act_qs[i * 4]);
            uint8_t b0 = qs[i * 4 + 0], b1 = qs[i * 4 + 1], b2 = qs[i * 4 + 2], b3 = qs[i * 4 + 3];
            int w_lo = (b0 & 0xF) | ((b1 & 0xF) << 8) | ((b2 & 0xF) << 16) | ((b3 & 0xF) << 24);
            int_sum = dp4a(a_packed, w_lo, int_sum);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int a_packed = *reinterpret_cast<const int*>(&act_qs[16 + i * 4]);
            uint8_t b0 = qs[i * 4 + 0], b1 = qs[i * 4 + 1], b2 = qs[i * 4 + 2], b3 = qs[i * 4 + 3];
            int w_hi = ((b0 >> 4) & 0xF) | (((b1 >> 4) & 0xF) << 8) |
                       (((b2 >> 4) & 0xF) << 16) | (((b3 >> 4) & 0xF) << 24);
            int_sum = dp4a(a_packed, w_hi, int_sum);
        }

        sum += w_scale * (act_scale * static_cast<float>(int_sum) - 8.0f * act_sum);
    }

    output[int64_t(m) * N + n] = sum;
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int64_t M, int64_t N, int64_t K) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    if (M == 1) {
        const int threads = 128;
        const int blocks = 128;
        gemm_warp_kernel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else if (M == 2) {
        const int threads = 256;
        const int blocks = 128;
        gemm_warp_kernel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else if (M <= 4) {
        const int threads = 256;
        const int blocks = 256;
        gemm_warp_kernel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else if (M <= 8) {
        const int threads = 256;
        const int blocks = 512;
        gemm_warp_kernel<<<blocks, threads>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    } else {
        const size_t shared_mem_size = NUM_K_BLOCKS * 2 * sizeof(float) + K * sizeof(int8_t);
        const int threads = 256;
        dim3 grid((N + threads - 1) / threads, M);
        dim3 block(threads);
        gemm_shared_kernel<<<grid, block, shared_mem_size>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 GEMM v19");
}
