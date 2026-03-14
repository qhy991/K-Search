/*
 * RMS Norm CUDA Kernel for Qwen2.5-7B - Optimized v6
 * hidden_size: 3584
 * Precision: FP32
 *
 * Based on v1 (105.4% of baseline) with further optimizations.
 *
 * Key optimizations:
 * 1. Float4 vectorization for memory bandwidth
 * 2. Adaptive thread count: 256 for batch<=8, 512 for batch>8
 * 3. Warp-level reduction with shuffle
 * 4. __ldg for read-only cache
 * 5. __fdividef and __frsqrt_rn for fast math
 * 6. Minimize register pressure
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr float EPSILON = 1e-6f;
constexpr int HIDDEN_SIZE = 3584;
constexpr int VEC_SIZE = 896;  // 3584 / 4

__forceinline__ __device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 256-thread kernel for small batches (better occupancy)
__global__ void __launch_bounds__(256) rms_norm_kernel_256(
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

    __shared__ float smem[8];
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < VEC_SIZE; i += 256) {
        float4 val = __ldg(&input_vec[i]);
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    float warp_sum = warp_reduce_sum(sum_sq);
    if (lane_id == 0) smem[warp_id] = warp_sum;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0 && lane_id < 8) total = smem[lane_id];
    if (warp_id == 0) {
        total = warp_reduce_sum(total);
        if (lane_id == 0) smem[0] = total;
    }
    __syncthreads();

    const float inv_rms = __fdividef(1.0f, sqrtf(__fdividef(smem[0], (float)HIDDEN_SIZE) + EPSILON));

    #pragma unroll 4
    for (int i = threadIdx.x; i < VEC_SIZE; i += 256) {
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

// 512-thread kernel for large batches (better throughput)
__global__ void __launch_bounds__(512) rms_norm_kernel_512(
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

    float sum_sq = 0.0f;
    #pragma unroll 2
    for (int i = threadIdx.x; i < VEC_SIZE; i += 512) {
        float4 val = __ldg(&input_vec[i]);
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    float warp_sum = warp_reduce_sum(sum_sq);
    if (lane_id == 0) smem[warp_id] = warp_sum;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0 && lane_id < 16) total = smem[lane_id];
    if (warp_id == 0) {
        total = warp_reduce_sum(total);
        if (lane_id == 0) smem[0] = total;
    }
    __syncthreads();

    const float inv_rms = __fdividef(1.0f, sqrtf(__fdividef(smem[0], (float)HIDDEN_SIZE) + EPSILON));

    #pragma unroll 2
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

    // Use 256 threads for batch <= 8 (better occupancy)
    // Use 512 threads for batch > 8 (better throughput)
    if (batch_size <= 8) {
        rms_norm_kernel_256<<<grid, 256>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    } else {
        rms_norm_kernel_512<<<grid, 512>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RMS Norm Qwen2.5-7B FP32 v6");
}
