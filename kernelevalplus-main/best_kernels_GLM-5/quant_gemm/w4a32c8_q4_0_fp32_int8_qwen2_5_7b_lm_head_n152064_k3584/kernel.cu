/**
 * W4A32C8 Quantized GEMM Kernel v16 for Qwen2.5-7B LM Head
 * - N: 152064 (output features / vocab size)
 * - K: 3584 (input features)
 * - Weight: Q4_0 quantized (4-bit with per-block scale, 18 bytes/block)
 * - Activation: FP32, dynamically quantized to Q8_1 style for INT8 compute
 *
 * V16: Combine best aspects from previous versions
 * - 2D grid for high occupancy (from v15)
 * - Warp K-parallelism (from final version)
 * - Shared activation quantization (from v15)
 * - Each warp processes outputs from its N_chunk
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int QK = 32;
constexpr int NUM_K_BLOCKS = 3584 / QK;  // 112
constexpr int Q4_0_BLOCK = 18;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 8

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

/**
 * Compute one block: weight Q4_0 x activation Q8_1
 */
__device__ __forceinline__ float compute_block_q4_0(
    const uint8_t* w_block,
    const int8_t* act_qs,
    float act_scale,
    float act_sum
) {
    float w_scale = half_to_float(*reinterpret_cast<const uint16_t*>(w_block));
    const uint8_t* qs = w_block + 2;

    int int_sum = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t packed = qs[i];
        int_sum += (packed & 0xF) * act_qs[i];
        int_sum += ((packed >> 4) & 0xF) * act_qs[i + 16];
    }

    return w_scale * (act_scale * static_cast<float>(int_sum) - 8.0f * act_sum);
}

/**
 * 2D grid kernel with warp K-parallelism
 * - blockIdx.y = M row
 * - blockIdx.x = N chunk
 * - Each warp processes outputs using K-parallelism
 */
template<int OUTPUTS_PER_WARP>
__global__ void __launch_bounds__(BLOCK_SIZE) gemm_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K,
    int N_per_block  // How many N values each block handles
) {
    const int m = blockIdx.y;
    const int n_chunk_start = blockIdx.x * N_per_block;

    if (m >= M) return;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_n_start = n_chunk_start + warp_id * OUTPUTS_PER_WARP;

    // Shared memory for activation quantization
    __shared__ float s_act_scales[112];
    __shared__ float s_act_sums[112];
    __shared__ int8_t s_act_qs[3584];

    const float* act_row = activation + (int64_t)m * K;

    // Cooperative quantization of activation (once per block)
    for (int kb = threadIdx.x; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_base = kb * QK;

        float a_max = 0.0f;
        float a_sum = 0.0f;
        float a_vals[QK];

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float val = __ldg(&act_row[k_base + i]);
            a_vals[i] = val;
            a_max = fmaxf(a_max, fabsf(val));
            a_sum += val;
        }

        const float scale = (a_max > 1e-10f) ? (a_max / 127.0f) : 1.0f;
        const float scale_inv = 127.0f / fmaxf(a_max, 1e-10f);

        s_act_scales[kb] = scale;
        s_act_sums[kb] = a_sum;

        #pragma unroll
        for (int i = 0; i < QK; i++) {
            float q = roundf(a_vals[i] * scale_inv);
            s_act_qs[k_base + i] = static_cast<int8_t>(
                fmaxf(-128.0f, fminf(127.0f, q)));
        }
    }
    __syncthreads();

    // Each warp processes OUTPUTS_PER_WARP outputs
    #pragma unroll
    for (int o = 0; o < OUTPUTS_PER_WARP; o++) {
        const int n = warp_n_start + o;
        if (n >= N) continue;

        // K-parallel: each lane processes a subset of K blocks
        float sum = 0.0f;

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const int k_base = kb * QK;
            const uint8_t* w_block = weight + ((int64_t)n * NUM_K_BLOCKS + kb) * Q4_0_BLOCK;

            sum += compute_block_q4_0(w_block, &s_act_qs[k_base], s_act_scales[kb], s_act_sums[kb]);
        }

        // Warp reduce
        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            output[(int64_t)m * N + n] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    // Use 2D grid: (N_chunks, M)
    // Each block has 8 warps, each warp handles 4 outputs
    // So each block handles 32 outputs
    constexpr int OUTPUTS_PER_WARP = 4;
    constexpr int N_per_block = WARPS_PER_BLOCK * OUTPUTS_PER_WARP;  // 32
    const int n_chunks = (N + N_per_block - 1) / N_per_block;

    dim3 grid(n_chunks, M);
    dim3 block(BLOCK_SIZE);

    gemm_kernel<OUTPUTS_PER_WARP><<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K,
        N_per_block
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Quantized GEMM Q4_0 Qwen2.5-7B LM Head v16");
}
