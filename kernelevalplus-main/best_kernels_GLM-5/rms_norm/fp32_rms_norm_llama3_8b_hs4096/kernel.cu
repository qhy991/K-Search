/*
 * RMS Norm CUDA Kernel for LLaMA-3-8B with hidden_size=4096 - Optimized v3
 *
 * Single warp per row approach with maximum optimization:
 * 1. One warp (32 threads) per row
 * 2. Float4 vectorization for coalesced memory access
 * 3. Full loop unrolling with #pragma unroll
 * 4. Warp-level reduction using __shfl_down_sync
 * 5. __frsqrt_rn for fast rsqrt computation
 * 6. __ldg for read-only cache
 * 7. Launch bounds to minimize register pressure
 *
 * hidden_size=4096 -> 1024 float4 -> 32 per thread
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr float EPSILON = 1e-6f;
constexpr int HIDDEN_SIZE = 4096;
constexpr int VEC_COUNT = HIDDEN_SIZE / 4;  // 1024 float4

__global__ void __launch_bounds__(512) rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size
) {
    // One warp per row
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= batch_size) return;

    const int lane = threadIdx.x;
    const float4* in4 = (const float4*)(input + row * HIDDEN_SIZE);
    float4* out4 = (float4*)(output + row * HIDDEN_SIZE);
    const float4* w4 = (const float4*)weight;

    // Sum of squares - 32 float4 per thread
    float ss = 0.0f;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float4 v = __ldg(&in4[lane + i * 32]);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Warp reduction
    ss += __shfl_down_sync(0xffffffff, ss, 16);
    ss += __shfl_down_sync(0xffffffff, ss, 8);
    ss += __shfl_down_sync(0xffffffff, ss, 4);
    ss += __shfl_down_sync(0xffffffff, ss, 2);
    ss += __shfl_down_sync(0xffffffff, ss, 1);

    // Broadcast inv_rms from lane 0
    float inv_rms = __shfl_sync(0xffffffff, __frsqrt_rn(ss / HIDDEN_SIZE + EPSILON), 0);

    // Normalize and scale
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int idx = lane + i * 32;
        float4 v = __ldg(&in4[idx]);
        float4 w = __ldg(&w4[idx]);
        out4[idx] = make_float4(
            v.x * inv_rms * w.x,
            v.y * inv_rms * w.y,
            v.z * inv_rms * w.z,
            v.w * inv_rms * w.w
        );
    }
}

torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kFloat32, "hidden_states must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");

    int batch_size = hidden_states.size(0);
    auto output = torch::empty({batch_size, HIDDEN_SIZE},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    // 16 warps per block = 16 rows per block
    dim3 block(32, 16);
    int grid = (batch_size + 15) / 16;

    rms_norm_kernel<<<grid, block>>>(
        hidden_states.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "FP32 RMS Norm LLaMA-3-8B hs4096 v3");
}
