/*
 * RMS Norm CUDA Kernel for DeepSeek-V3 - Optimized v2
 * hidden_size: 7168
 * Precision: FP32
 *
 * Optimization strategies:
 * 1. Increased ILP with wider unrolling
 * 2. Better memory coalescing with aligned access
 * 3. Reduced shared memory bank conflicts
 * 4. Optimized warp reduction
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr float EPSILON = 1e-6f;
constexpr int HIDDEN_SIZE = 7168;
constexpr int VEC_SIZE = HIDDEN_SIZE / 4;  // 1792 float4 elements

// Optimized warp reduction using shuffle intrinsics
__forceinline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction using shared memory
__forceinline__ __device__ float block_reduce_sum(float val, float* shared) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // Warp reduction
    val = warp_reduce_sum(val);

    // First lane of each warp writes to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces shared memory
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

// Version with 256 threads - each thread processes more elements
__global__ void __launch_bounds__(256) rms_norm_256(
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

    __shared__ float shared_sums[8];  // 256 / 32 = 8 warps

    const float4* __restrict__ input_vec = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ weight_vec = reinterpret_cast<const float4*>(weight);
    float4* __restrict__ out_vec = reinterpret_cast<float4*>(out);

    // Phase 1: Sum of squares with vectorized loads
    float local_sum_sq = 0.0f;

    // Process 7 elements per thread (1792 / 256 = 7)
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 val = __ldg(&input_vec[idx]);
            local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        }
    }

    // Phase 2: Block reduction
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_sums);

    // Broadcast result to all threads
    if (threadIdx.x == 0) {
        shared_sums[0] = total_sum_sq;
    }
    __syncthreads();
    total_sum_sq = shared_sums[0];

    // Phase 3: Normalize and scale
    const float inv_rms = __frsqrt_rn(total_sum_sq * (1.0f / (float)hidden_size) + EPSILON);

    #pragma unroll
    for (int i = 0; i < 7; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 in_val = __ldg(&input_vec[idx]);
            const float4 w_val = __ldg(&weight_vec[idx]);
            float4 out_val;
            out_val.x = in_val.x * inv_rms * w_val.x;
            out_val.y = in_val.y * inv_rms * w_val.y;
            out_val.z = in_val.z * inv_rms * w_val.z;
            out_val.w = in_val.w * inv_rms * w_val.w;
            out_vec[idx] = out_val;
        }
    }
}

// Version with 512 threads - balanced approach
__global__ void __launch_bounds__(512) rms_norm_512(
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

    __shared__ float shared_sums[16];  // 512 / 32 = 16 warps

    const float4* __restrict__ input_vec = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ weight_vec = reinterpret_cast<const float4*>(weight);
    float4* __restrict__ out_vec = reinterpret_cast<float4*>(out);

    // Phase 1: Sum of squares with vectorized loads
    float local_sum_sq = 0.0f;

    // Process 4 elements per thread (1792 / 512 = 3.5, round up to 4)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 val = __ldg(&input_vec[idx]);
            local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        }
    }

    // Phase 2: Block reduction
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_sums);

    // Broadcast result to all threads
    if (threadIdx.x == 0) {
        shared_sums[0] = total_sum_sq;
    }
    __syncthreads();
    total_sum_sq = shared_sums[0];

    // Phase 3: Normalize and scale
    const float inv_rms = __frsqrt_rn(total_sum_sq * (1.0f / (float)hidden_size) + EPSILON);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 in_val = __ldg(&input_vec[idx]);
            const float4 w_val = __ldg(&weight_vec[idx]);
            float4 out_val;
            out_val.x = in_val.x * inv_rms * w_val.x;
            out_val.y = in_val.y * inv_rms * w_val.y;
            out_val.z = in_val.z * inv_rms * w_val.z;
            out_val.w = in_val.w * inv_rms * w_val.w;
            out_vec[idx] = out_val;
        }
    }
}

// Version with 1024 threads for better occupancy at larger batches
__global__ void __launch_bounds__(1024) rms_norm_1024(
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

    __shared__ float shared_sums[32];  // 1024 / 32 = 32 warps

    const float4* __restrict__ input_vec = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ weight_vec = reinterpret_cast<const float4*>(weight);
    float4* __restrict__ out_vec = reinterpret_cast<float4*>(out);

    // Phase 1: Sum of squares with vectorized loads
    float local_sum_sq = 0.0f;

    // Process 2 elements per thread (1792 / 1024 = 1.75, round up to 2)
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 val = __ldg(&input_vec[idx]);
            local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        }
    }

    // Phase 2: Block reduction
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_sums);

    // Broadcast result to all threads
    if (threadIdx.x == 0) {
        shared_sums[0] = total_sum_sq;
    }
    __syncthreads();
    total_sum_sq = shared_sums[0];

    // Phase 3: Normalize and scale
    const float inv_rms = __frsqrt_rn(total_sum_sq * (1.0f / (float)hidden_size) + EPSILON);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < VEC_SIZE) {
            const float4 in_val = __ldg(&input_vec[idx]);
            const float4 w_val = __ldg(&weight_vec[idx]);
            float4 out_val;
            out_val.x = in_val.x * inv_rms * w_val.x;
            out_val.y = in_val.y * inv_rms * w_val.y;
            out_val.z = in_val.z * inv_rms * w_val.z;
            out_val.w = in_val.w * inv_rms * w_val.w;
            out_vec[idx] = out_val;
        }
    }
}

// PyTorch interface with dynamic dispatch
torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    const int batch_size = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);

    auto output = torch::empty({batch_size, hidden_size},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    dim3 grid(batch_size);

    // Dispatch based on batch size for optimal performance
    if (batch_size <= 4) {
        // Smaller batch: use 1024 threads for better memory throughput
        dim3 block(1024);
        rms_norm_1024<<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    } else if (batch_size <= 64) {
        // Medium batch: use 512 threads
        dim3 block(512);
        rms_norm_512<<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    } else {
        // Large batch: use 256 threads for better occupancy
        dim3 block(256);
        rms_norm_256<<<grid, block>>>(
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
    m.def("forward", &forward, "RMS Norm Optimized v2");
}
