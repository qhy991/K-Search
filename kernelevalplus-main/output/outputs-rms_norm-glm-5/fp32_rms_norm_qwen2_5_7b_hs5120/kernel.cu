/*
 * RMS Norm CUDA Kernel for Qwen2.5-7B - Final Optimized Version
 * hidden_size: 5120
 * Precision: FP32
 *
 * Key optimizations:
 * 1. Float4 vectorization for memory bandwidth
 * 2. Optimized warp reduction with shuffle intrinsics
 * 3. __ldg for read-only cache optimization
 * 4. __frsqrt_rn for fast reciprocal sqrt
 * 5. Pre-computed reciprocal of hidden_size
 * 6. Loop unrolling for better ILP
 * 7. __launch_bounds__ for optimal occupancy
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr float EPSILON = 1e-6f;
constexpr int VEC_SIZE = 1280;  // 5120 / 4
constexpr float INV_HIDDEN_SIZE = 1.0f / 5120.0f;

__forceinline__ __device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 512-thread kernel optimized for hidden_size=5120
__global__ void __launch_bounds__(512, 2) rms_norm_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float4* __restrict__ input_vec = (const float4*)(hidden_states + batch_idx * hidden_size);
    float4* __restrict__ out_vec = (float4*)(output + batch_idx * hidden_size);
    const float4* __restrict__ weight_vec = (const float4*)weight;

    __shared__ float smem[16];
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // Phase 1: Compute sum of squares with vectorized loads
    float sum_sq = 0.0f;

    #pragma unroll 3
    for (int i = threadIdx.x; i < VEC_SIZE; i += 512) {
        float4 val = __ldg(&input_vec[i]);
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // Phase 2: Block reduction
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0 && lane_id < 16) total = smem[lane_id];
    if (warp_id == 0) total = warp_reduce_sum(total);
    if (threadIdx.x == 0) smem[0] = total;
    __syncthreads();

    // Phase 3: Normalize and apply weight
    const float inv_rms = __frsqrt_rn(smem[0] * INV_HIDDEN_SIZE + EPSILON);

    #pragma unroll 3
    for (int i = threadIdx.x; i < VEC_SIZE; i += 512) {
        float4 in_val = __ldg(&input_vec[i]);
        float4 w_val = __ldg(&weight_vec[i]);
        out_vec[i] = make_float4(
            in_val.x * inv_rms * w_val.x,
            in_val.y * inv_rms * w_val.y,
            in_val.z * inv_rms * w_val.z,
            in_val.w * inv_rms * w_val.w
        );
    }
}

torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    const int batch_size = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);

    auto output = torch::empty({batch_size, hidden_size},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    const dim3 grid(batch_size);
    const dim3 block(512);

    rms_norm_kernel<<<grid, block>>>(
        hidden_states.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RMS Norm Qwen2.5-7B FP32 hs5120 Final");
}
