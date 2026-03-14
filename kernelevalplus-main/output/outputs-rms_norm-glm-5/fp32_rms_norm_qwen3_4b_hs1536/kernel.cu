/*
 * RMS Norm CUDA Kernel for Qwen3-4B - Optimized v3
 * hidden_size: 1536
 * Precision: FP32
 *
 * Performance optimizations:
 * 1. Aggressive loop unrolling
 * 2. Use L1 cache for weight (read-only)
 * 3. Minimize register pressure
 * 4. Optimized for minimum kernel launch overhead
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr float EPSILON = 1e-6f;
constexpr int HIDDEN_SIZE = 1536;

// Single-warp kernel - processes all 384 float4 elements with 32 threads
// Each thread handles 12 float4 elements
__global__ void __launch_bounds__(32, 16) rms_norm_warp32(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float4* __restrict__ in = (const float4*)(hidden_states + batch_idx * hidden_size);
    const float4* __restrict__ w = (const float4*)weight;
    float4* __restrict__ out = (float4*)(output + batch_idx * hidden_size);

    const int tid = threadIdx.x;
    float sum = 0.0f;

    // Unrolled processing of 12 elements per thread
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        const int idx = tid + i * 32;
        const float4 v = in[idx];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Warp reduction - no shared memory needed
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Broadcast inv_rms to all threads
    const float inv_rms = __frsqrt_rn(sum / (float)hidden_size + EPSILON);

    // Output phase
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        const int idx = tid + i * 32;
        const float4 v = in[idx];
        const float4 g = __ldg(&w[idx]);
        float4 r;
        r.x = v.x * inv_rms * g.x;
        r.y = v.y * inv_rms * g.y;
        r.z = v.z * inv_rms * g.z;
        r.w = v.w * inv_rms * g.w;
        out[idx] = r;
    }
}

// 64-thread kernel - processes all elements with 2 warps
__global__ void __launch_bounds__(64, 12) rms_norm_warp64(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float4* __restrict__ in = (const float4*)(hidden_states + batch_idx * hidden_size);
    const float4* __restrict__ w = (const float4*)weight;
    float4* __restrict__ out = (float4*)(output + batch_idx * hidden_size);

    const int tid = threadIdx.x;
    float sum = 0.0f;

    // 6 elements per thread (384 / 64 = 6)
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        const int idx = tid + i * 64;
        if (idx < 384) {
            const float4 v = in[idx];
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s[2];
    if ((tid & 31) == 0) {
        s[tid >> 5] = sum;
    }
    __syncthreads();

    const float total = s[0] + s[1];
    const float inv_rms = __frsqrt_rn(total / (float)hidden_size + EPSILON);

    // Output
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        const int idx = tid + i * 64;
        if (idx < 384) {
            const float4 v = in[idx];
            const float4 g = __ldg(&w[idx]);
            float4 r;
            r.x = v.x * inv_rms * g.x;
            r.y = v.y * inv_rms * g.y;
            r.z = v.z * inv_rms * g.z;
            r.w = v.w * inv_rms * g.w;
            out[idx] = r;
        }
    }
}

// 128-thread kernel - most balanced for general use
__global__ void __launch_bounds__(128, 8) rms_norm_warp128(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float4* __restrict__ in = (const float4*)(hidden_states + batch_idx * hidden_size);
    const float4* __restrict__ w = (const float4*)weight;
    float4* __restrict__ out = (float4*)(output + batch_idx * hidden_size);

    const int tid = threadIdx.x;
    float sum = 0.0f;

    // 3 elements per thread (384 / 128 = 3)
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const int idx = tid + i * 128;
        const float4 v = in[idx];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduction
    __shared__ float s[4];
    if ((tid & 31) == 0) {
        s[tid >> 5] = sum;
    }
    __syncthreads();

    float total = 0.0f;
    if (tid < 32) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (tid + i * 32 < 4) total += s[tid + i * 32];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
    }
    __syncthreads();

    // Broadcast
    __shared__ float inv_rms_shared;
    if (tid == 0) {
        inv_rms_shared = __frsqrt_rn(total / (float)hidden_size + EPSILON);
    }
    __syncthreads();
    const float inv_rms = inv_rms_shared;

    // Output
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const int idx = tid + i * 128;
        const float4 v = in[idx];
        const float4 g = __ldg(&w[idx]);
        float4 r;
        r.x = v.x * inv_rms * g.x;
        r.y = v.y * inv_rms * g.y;
        r.z = v.z * inv_rms * g.z;
        r.w = v.w * inv_rms * g.w;
        out[idx] = r;
    }
}

// 384-thread kernel - 1 float4 per thread, best for large batches
__global__ void __launch_bounds__(384, 4) rms_norm_warp384(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float4* __restrict__ in = (const float4*)(hidden_states + batch_idx * hidden_size);
    const float4* __restrict__ w = (const float4*)weight;
    float4* __restrict__ out = (float4*)(output + batch_idx * hidden_size);

    const int tid = threadIdx.x;
    float sum = 0.0f;

    // 1 element per thread
    if (tid < 384) {
        const float4 v = in[tid];
        sum = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduction (12 warps)
    __shared__ float s[12];
    if ((tid & 31) == 0) {
        s[tid >> 5] = sum;
    }
    __syncthreads();

    float total = 0.0f;
    if (tid < 32) {
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            if (tid + i * 32 < 12) total += s[tid + i * 32];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
    }
    __syncthreads();

    // Broadcast
    __shared__ float inv_rms_shared;
    if (tid == 0) {
        inv_rms_shared = __frsqrt_rn(total / (float)hidden_size + EPSILON);
    }
    __syncthreads();
    const float inv_rms = inv_rms_shared;

    // Output
    if (tid < 384) {
        const float4 v = in[tid];
        const float4 g = __ldg(&w[tid]);
        float4 r;
        r.x = v.x * inv_rms * g.x;
        r.y = v.y * inv_rms * g.y;
        r.z = v.z * inv_rms * g.z;
        r.w = v.w * inv_rms * g.w;
        out[tid] = r;
    }
}

// PyTorch interface
torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    const int batch_size = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);

    auto output = torch::empty({batch_size, hidden_size},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    dim3 grid(batch_size);

    // Optimal dispatch based on batch size
    // For small batches, fewer threads = less overhead
    if (batch_size <= 2) {
        dim3 block(32);
        rms_norm_warp32<<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    } else if (batch_size <= 8) {
        dim3 block(64);
        rms_norm_warp64<<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    } else if (batch_size <= 128) {
        dim3 block(128);
        rms_norm_warp128<<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size
        );
    } else {
        dim3 block(384);
        rms_norm_warp384<<<grid, block>>>(
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
    m.def("forward", &forward, "RMS Norm for Qwen3-4B (hidden_size=1536) v3");
}
