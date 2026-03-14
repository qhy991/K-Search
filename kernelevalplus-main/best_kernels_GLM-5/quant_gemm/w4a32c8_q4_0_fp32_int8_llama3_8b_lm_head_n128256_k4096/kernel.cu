/**
 * W4A32C8 Quantized GEMM Kernel v11 for LLaMA-3-8B LM Head
 * - N: 128256 (output features / vocab size)
 * - K: 4096 (input features)
 * - Weight: Q4_0 quantized (4-bit with per-block scale, 18 bytes/block)
 * - Activation: FP32, dynamically quantized to Q8_1 style for INT8 compute
 *
 * V11 OPTIMIZATIONS:
 * 1. Use __ldg for explicit read-only cache hints on weights
 * 2. Process outputs in groups that share K-block access patterns
 * 3. Use shared memory for activation quantization (cooperative)
 * 4. Minimize register pressure
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int K = 4096;
constexpr int NUM_K_BLOCKS = K / QK;  // 128
constexpr int Q4_0_BLOCK = 18;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 8
constexpr int OUTPUTS_PER_WARP = 4;

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Load weight block using __ldg for read-only cache
 */
__device__ __forceinline__ void load_weight_block_ldg(const uint8_t* ptr, uint8_t* buffer) {
    // Load 18 bytes using __ldg for L1 cache hint
    const uint4* ptr4 = reinterpret_cast<const uint4*>(ptr);
    uint4 data = __ldg(ptr4);
    *reinterpret_cast<uint4*>(buffer) = data;
    // Load remaining 2 bytes
    const ushort* ptr2 = reinterpret_cast<const ushort*>(ptr + 16);
    *reinterpret_cast<ushort*>(buffer + 16) = __ldg(ptr2);
}

/**
 * Compute one Q4_0 block contribution
 */
__device__ __forceinline__ float compute_block(
    const uint8_t* w_block,
    const float* act_ptr
) {
    // Load weight scale using __ldg
    float d_w = half_to_float(__ldg(reinterpret_cast<const uint16_t*>(w_block)));

    float a_vals[QK];
    float a_max = 0.0f;
    float a_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < QK; i += 4) {
        float4 v = *reinterpret_cast<const float4*>(act_ptr + i);
        a_vals[i] = v.x; a_vals[i+1] = v.y; a_vals[i+2] = v.z; a_vals[i+3] = v.w;
    }

    #pragma unroll
    for (int i = 0; i < QK; i++) {
        a_max = fmaxf(a_max, fabsf(a_vals[i]));
        a_sum += a_vals[i];
    }

    float d_a = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;
    float scale = 127.0f / fmaxf(a_max, 1e-10f);

    const uint8_t* qs = w_block + 2;
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

/**
 * Strategy 1: Warp-level K-parallelism (best for small M)
 */
__global__ void __launch_bounds__(BLOCK_SIZE) gemm_warp_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_warps = gridDim.x * WARPS_PER_BLOCK;

    for (int base = global_warp_id * OUTPUTS_PER_WARP; base < M * N; base += total_warps * OUTPUTS_PER_WARP) {
        int m_idx[OUTPUTS_PER_WARP], n_idx[OUTPUTS_PER_WARP];
        float sums[OUTPUTS_PER_WARP] = {0.0f};

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            int idx = base + o;
            m_idx[o] = idx / N;
            n_idx[o] = idx % N;
            if (idx >= M * N) m_idx[o] = -1;
        }

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            int k_start = kb * QK;

            #pragma unroll
            for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
                if (m_idx[o] < 0) continue;

                const uint8_t* w_block = weight + (int64_t(n_idx[o]) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;
                const float* a_ptr = activation + m_idx[o] * K + k_start;
                sums[o] += compute_block(w_block, a_ptr);
            }
        }

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
            sums[o] = warp_reduce_sum(sums[o]);
            if (lane_id == 0 && m_idx[o] >= 0) {
                output[m_idx[o] * N + n_idx[o]] = sums[o];
            }
        }
    }
}

/**
 * Strategy 2: Per-thread output with shared memory (large M)
 */
__global__ void __launch_bounds__(BLOCK_SIZE) gemm_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    // Shared memory for activation quantization
    extern __shared__ char shared_mem[];
    float* s_act_scales = reinterpret_cast<float*>(shared_mem);
    float* s_act_sums = reinterpret_cast<float*>(shared_mem + NUM_K_BLOCKS * sizeof(float));
    int8_t* s_act_qs = reinterpret_cast<int8_t*>(shared_mem + NUM_K_BLOCKS * 2 * sizeof(float));

    const float* act_row = activation + m * K;

    // Cooperative quantization
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
        const int k_base = kb * QK;
        const float act_scale = s_act_scales[kb];
        const float act_sum = s_act_sums[kb];
        const int8_t* act_qs = &s_act_qs[k_base];

        const uint8_t* w_block = weight + (int64_t(n) * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;

        float w_scale;
        int8_t w_qs[QK];
        w_scale = half_to_float(__ldg(reinterpret_cast<const uint16_t*>(w_block)));
        const uint8_t* qs = w_block + 2;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = __ldg(&qs[i]);
            w_qs[i] = static_cast<int8_t>(packed & 0x0F);
            w_qs[i + 16] = static_cast<int8_t>((packed >> 4) & 0x0F);
        }

        int int_sum = 0;
        #pragma unroll
        for (int i = 0; i < QK; i += 4) {
            const int a_packed = *reinterpret_cast<const int*>(&act_qs[i]);
            const int w_packed = *reinterpret_cast<const int*>(&w_qs[i]);
            int_sum = dp4a(a_packed, w_packed, int_sum);
        }

        sum += w_scale * (act_scale * static_cast<float>(int_sum) - 8.0f * act_sum);
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    if (M <= 8) {
        constexpr int OUTPUTS_PER_BLOCK = WARPS_PER_BLOCK * OUTPUTS_PER_WARP;
        int total_outputs = M * N;
        int num_blocks = (total_outputs + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK;
        num_blocks = min(num_blocks, 65536);

        dim3 grid(num_blocks);
        dim3 block(BLOCK_SIZE);

        gemm_warp_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        const size_t shared_mem_size = NUM_K_BLOCKS * 2 * sizeof(float) + K * sizeof(int8_t);

        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
        dim3 block(BLOCK_SIZE);

        gemm_shared_kernel<<<grid, block, shared_mem_size>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM Q4_0 LM Head v11");
}
