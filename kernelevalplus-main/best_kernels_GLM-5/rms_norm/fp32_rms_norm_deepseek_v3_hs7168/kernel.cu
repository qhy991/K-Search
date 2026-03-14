/*
 * RMS Norm CUDA Kernel for DeepSeek-V3 - Optimized v3
 * hidden_size: 7168
 * Precision: FP32
 *
 * Key optimizations:
 * 1. Float4 vectorization for memory bandwidth
 * 2. Warp-level reduction with __shfl_down_sync
 * 3. __ldg for read-only cache hints
 * 4. __fdividef for faster division
 * 5. Multiple element processing per thread for better ILP
 * 6. Optimized for RTX 4090 architecture (Ada Lovelace)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr float EPSILON = 1e-6f;

// Warp-level reduction using shuffle instructions
__forceinline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// High-performance kernel with loop unrolling
__global__ void __launch_bounds__(512) rms_norm_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* __restrict__ input = hidden_states + batch_idx * hidden_size;
    float* __restrict__ out = output + batch_idx * hidden_size;

    __shared__ float shared_sum[16];

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int vec_elements = hidden_size >> 2;  // 1792 float4 elements

    // Phase 1: Compute sum of squares
    float local_sum_sq = 0.0f;
    const float4* __restrict__ input_vec = reinterpret_cast<const float4*>(input);

    // Process multiple elements per iteration for better ILP
    // With 512 threads and 1792 elements, each thread processes 3-4 elements
    // Unroll by 4 for maximum throughput
    #pragma unroll
    for (int i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        const float4 val = __ldg(&input_vec[i]);
        local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // Warp reduction
    float warp_sum = warp_reduce_sum(local_sum_sq);

    if (lane_id == 0) {
        shared_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction
    float total_sum_sq = 0.0f;
    if (warp_id == 0 && lane_id < 16) {
        total_sum_sq = shared_sum[lane_id];
    }
    if (warp_id == 0) {
        total_sum_sq = warp_reduce_sum(total_sum_sq);
        if (lane_id == 0) {
            shared_sum[0] = total_sum_sq;
        }
    }
    __syncthreads();

    total_sum_sq = shared_sum[0];
    const float inv_rms = __fdividef(1.0f, sqrtf(__fdividef(total_sum_sq, (float)hidden_size) + EPSILON));

    // Phase 2: Normalize and write output
    float4* __restrict__ out_vec = reinterpret_cast<float4*>(out);
    const float4* __restrict__ weight_vec = reinterpret_cast<const float4*>(weight);

    #pragma unroll
    for (int i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        const float4 in_val = __ldg(&input_vec[i]);
        const float4 w_val = __ldg(&weight_vec[i]);
        float4 out_val;
        out_val.x = in_val.x * inv_rms * w_val.x;
        out_val.y = in_val.y * inv_rms * w_val.y;
        out_val.z = in_val.z * inv_rms * w_val.z;
        out_val.w = in_val.w * inv_rms * w_val.w;
        out_vec[i] = out_val;
    }
}

// Smaller block kernel for better occupancy with small batches
__global__ void __launch_bounds__(256) rms_norm_kernel_small(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* __restrict__ input = hidden_states + batch_idx * hidden_size;
    float* __restrict__ out = output + batch_idx * hidden_size;

    __shared__ float shared_sum[8];

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int vec_elements = hidden_size >> 2;

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    const float4* __restrict__ input_vec = reinterpret_cast<const float4*>(input);

    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        const float4 val = __ldg(&input_vec[i]);
        local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // Warp reduction
    float warp_sum = warp_reduce_sum(local_sum_sq);

    if (lane_id == 0) {
        shared_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction
    float total_sum_sq = 0.0f;
    if (warp_id == 0 && lane_id < 8) {
        total_sum_sq = shared_sum[lane_id];
    }
    if (warp_id == 0) {
        total_sum_sq = warp_reduce_sum(total_sum_sq);
        if (lane_id == 0) {
            shared_sum[0] = total_sum_sq;
        }
    }
    __syncthreads();

    total_sum_sq = shared_sum[0];
    const float inv_rms = __fdividef(1.0f, sqrtf(__fdividef(total_sum_sq, (float)hidden_size) + EPSILON));

    // Normalize
    float4* __restrict__ out_vec = reinterpret_cast<float4*>(out);
    const float4* __restrict__ weight_vec = reinterpret_cast<const float4*>(weight);

    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        const float4 in_val = __ldg(&input_vec[i]);
        const float4 w_val = __ldg(&weight_vec[i]);
        float4 out_val;
        out_val.x = in_val.x * inv_rms * w_val.x;
        out_val.y = in_val.y * inv_rms * w_val.y;
        out_val.z = in_val.z * inv_rms * w_val.z;
        out_val.w = in_val.w * inv_rms * w_val.w;
        out_vec[i] = out_val;
    }
}

torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    const int batch_size = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);

    auto output = torch::empty({batch_size, hidden_size},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    dim3 grid(batch_size);

    // For batch_size <= 8, use smaller blocks for better occupancy
    // For batch_size > 8, use larger blocks for throughput
    if (batch_size <= 8) {
        dim3 block(256);
        rms_norm_kernel_small<<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    } else {
        dim3 block(512);
        rms_norm_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "RMS Norm DeepSeek-V3 FP32 v3");
}
