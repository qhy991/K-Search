/*
 * RMS Norm CUDA Kernel for LLaMA2-7B hidden_size=128 - Final Optimized
 *
 * Optimized for NVIDIA RTX 4090 (Ada Lovelace architecture)
 *
 * Key optimizations:
 * 1. Float4 vectorization for memory bandwidth (128 = 32 * 4)
 * 2. Warp-per-row with shuffle reduction
 * 3. __ldg for read-only cache hints
 * 4. __fdividef for faster division
 * 5. Launch bounds optimized for occupancy
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr float EPSILON = 1e-6f;
constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 128;  // Fixed for this kernel variant

/**
 * Optimized Warp-per-row RMS Norm Kernel
 *
 * For hidden_size=128:
 * - Each warp (32 threads) processes one row
 * - Each thread loads 1 float4 (4 elements) using vectorized load
 * - Full unrolling for maximum ILP
 */
__global__ void __launch_bounds__(256) rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size
) {
    const int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    if (warp_id >= batch_size) return;

    const int lane_id = threadIdx.x;

    // Vectorized pointers
    const float4* row_in_vec = reinterpret_cast<const float4*>(input + warp_id * HIDDEN_SIZE);
    float4* row_out_vec = reinterpret_cast<float4*>(output + warp_id * HIDDEN_SIZE);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);

    // Load input and weight using float4
    const float4 val = __ldg(&row_in_vec[lane_id]);
    const float4 w = __ldg(&weight_vec[lane_id]);

    // Compute sum of squares
    float sum_sq = val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;

    // Warp-level reduction using shuffle intrinsics
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Compute inv_rms (broadcast from lane 0)
    const float mean_sq = __shfl_sync(0xffffffff, sum_sq, 0) / HIDDEN_SIZE;
    const float inv_rms = __fdividef(1.0f, sqrtf(mean_sq + EPSILON));

    // Normalize and apply weight
    float4 out_val;
    out_val.x = val.x * inv_rms * w.x;
    out_val.y = val.y * inv_rms * w.y;
    out_val.z = val.z * inv_rms * w.z;
    out_val.w = val.w * inv_rms * w.w;

    // Store output
    row_out_vec[lane_id] = out_val;
}

/**
 * Host entry point
 */
torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kFloat32, "hidden_states must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");

    const int batch_size = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);

    TORCH_CHECK(hidden_size == HIDDEN_SIZE, "hidden_size must be 128 for this kernel");

    auto output = torch::empty({batch_size, hidden_size},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    const float* input_ptr = hidden_states.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Use up to 8 warps per block for optimal occupancy
    int warps_per_block = min(batch_size, 8);
    dim3 block(WARP_SIZE, warps_per_block);
    const int blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    rms_norm_kernel<<<blocks, block>>>(
        input_ptr, weight_ptr, output_ptr, batch_size
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "FP32 RMS Norm for LLaMA2-7B hidden_size=128 Final");
}
