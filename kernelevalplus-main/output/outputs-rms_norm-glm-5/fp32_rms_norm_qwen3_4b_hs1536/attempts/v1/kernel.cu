/*
 * RMS Norm CUDA Kernel for Qwen3-4B - Version 1 (Initial)
 * hidden_size: 1536
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr float EPSILON = 1e-6f;
constexpr int HIDDEN_SIZE = 1536;
constexpr int VEC_SIZE = HIDDEN_SIZE / 4;

__forceinline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__forceinline__ __device__ float block_reduce_sum(float val, float* shared) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    val = 0.0f;
    if (warp_id == 0) {
        const int num_warps = blockDim.x >> 5;
        if (lane_id < num_warps) {
            val = shared[lane_id];
        }
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;
}

__global__ void rms_norm_128(const float* hidden_states, const float* weight,
                              float* output, int batch_size, int hidden_size) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* input = hidden_states + batch_idx * hidden_size;
    float* out = output + batch_idx * hidden_size;
    __shared__ float shared_sums[4];
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* out_vec = reinterpret_cast<float4*>(out);
    
    float local_sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 val = input_vec[idx];
            local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        }
    }
    
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_sums);
    if (threadIdx.x == 0) shared_sums[0] = total_sum_sq;
    __syncthreads();
    total_sum_sq = shared_sums[0];
    
    const float inv_rms = __frsqrt_rn(total_sum_sq * (1.0f / (float)hidden_size) + EPSILON);
    
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 in_val = input_vec[idx];
            const float4 w_val = weight_vec[idx];
            float4 out_val;
            out_val.x = in_val.x * inv_rms * w_val.x;
            out_val.y = in_val.y * inv_rms * w_val.y;
            out_val.z = in_val.z * inv_rms * w_val.z;
            out_val.w = in_val.w * inv_rms * w_val.w;
            out_vec[idx] = out_val;
        }
    }
}

torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    const int batch_size = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);
    auto output = torch::empty({batch_size, hidden_size},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));
    dim3 grid(batch_size);
    dim3 block(128);
    rms_norm_128<<<grid, block>>>(
        hidden_states.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, hidden_size
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RMS Norm v1");
}
