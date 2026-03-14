#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

constexpr float EPSILON = 1e-6f;
constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 512;

/**
 * Optimization v5: Best combination
 *
 * - Warp-per-row (optimal for all batch sizes)
 * - Vectorized float4 loads/stores
 * - rsqrt instead of sqrt + divide
 * - Full unrolling
 */
__global__ void __launch_bounds__(512) rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size
) {
    // One warp per row
    const int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    if (warp_id >= batch_size) return;

    const int lane_id = threadIdx.x;

    // Vectorized access
    const float4* input4 = reinterpret_cast<const float4*>(input + warp_id * HIDDEN_SIZE);
    float4* output4 = reinterpret_cast<float4*>(output + warp_id * HIDDEN_SIZE);
    const float4* weight4 = reinterpret_cast<const float4*>(weight);

    // Each thread handles 4 vectors (16 floats total)
    // 128 vectors / 32 threads = 4 vectors per thread
    float sum_sq = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = lane_id + i * WARP_SIZE;
        const float4 val = input4[idx];
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Compute inv_rms using rsqrt
    const float inv_rms = __shfl_sync(0xffffffff, __frsqrt_rn(sum_sq / HIDDEN_SIZE + EPSILON), 0);

    // Normalize and scale
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = lane_id + i * WARP_SIZE;
        const float4 val = input4[idx];
        const float4 w = weight4[idx];
        float4 out;
        out.x = val.x * inv_rms * w.x;
        out.y = val.y * inv_rms * w.y;
        out.z = val.z * inv_rms * w.z;
        out.w = val.w * inv_rms * w.w;
        output4[idx] = out;
    }
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

    auto output = torch::empty({batch_size, HIDDEN_SIZE},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    const float* input_ptr = hidden_states.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Warp-per-row: 16 warps per block (16 rows per block)
    dim3 block(WARP_SIZE, 16);  // 512 threads
    const int blocks = (batch_size + 15) / 16;

    rms_norm_kernel<<<blocks, block>>>(
        input_ptr, weight_ptr, output_ptr, batch_size
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "FP32 RMS Norm for LLaMA2-7B hidden_size=512");
}
