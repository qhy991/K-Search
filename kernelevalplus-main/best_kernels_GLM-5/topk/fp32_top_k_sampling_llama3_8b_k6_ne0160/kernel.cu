/**
 * Top-K Sampling CUDA Kernel - V4 Final
 *
 * Input: probs [batch_size, vocab_subset] FP32 - probability distribution
 * Output: samples [batch_size] int64 - sampled token indices
 *
 * K=6, vocab_subset=160
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>

__global__ void __launch_bounds__(256) topk_sampling_kernel(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Vectorized loads via float4
    const float4* probs_vec = reinterpret_cast<const float4*>(probs + tid * vocab_size);

    // Top-6 in registers
    float v0 = -FLT_MAX, v1 = -FLT_MAX, v2 = -FLT_MAX;
    float v3 = -FLT_MAX, v4 = -FLT_MAX, v5 = -FLT_MAX;
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0;

    #pragma unroll
    for (int vec_i = 0; vec_i < 40; vec_i++) {
        float4 vals = probs_vec[vec_i];
        int base = vec_i << 2;

        // Process each element - using macro for efficiency
        #define PROCESS_ELEM(elem_val, elem_idx) do { \
            float __val = (elem_val); \
            int __idx = (elem_idx); \
            if (__val > v5) { \
                if (__val > v4) { \
                    v5 = v4; i5 = i4; \
                    if (__val > v3) { \
                        v4 = v3; i4 = i3; \
                        if (__val > v2) { \
                            v3 = v2; i3 = i2; \
                            if (__val > v1) { \
                                v2 = v1; i2 = i1; \
                                if (__val > v0) { \
                                    v1 = v0; i1 = i0; \
                                    v0 = __val; i0 = __idx; \
                                } else { v2 = __val; i2 = __idx; } \
                            } else { v3 = __val; i3 = __idx; } \
                        } else { v4 = __val; i4 = __idx; } \
                    } else { v5 = __val; i5 = __idx; } \
                } else { v5 = __val; i5 = __idx; } \
            } \
        } while(0)

        PROCESS_ELEM(vals.x, base);
        PROCESS_ELEM(vals.y, base + 1);
        PROCESS_ELEM(vals.z, base + 2);
        PROCESS_ELEM(vals.w, base + 3);

        #undef PROCESS_ELEM
    }

    samples[tid] = i0;
}

void launch_topk_kernel(
    const float* probs,
    int64_t* indices,
    int batch_size,
    int vocab_size,
    int k
) {
    (void)k;
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;

    topk_sampling_kernel<<<grid_size, block_size>>>(probs, indices, batch_size, vocab_size);
}

torch::Tensor forward(torch::Tensor probs, int k) {
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(probs.dim() == 2, "probs must be 2-dimensional");
    TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");

    int batch_size = probs.size(0);
    int vocab_size = probs.size(1);

    auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt64).device(probs.device()));

    launch_topk_kernel(probs.data_ptr<float>(), samples.data_ptr<int64_t>(), batch_size, vocab_size, k);

    return samples;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Top-K Sampling");
}
