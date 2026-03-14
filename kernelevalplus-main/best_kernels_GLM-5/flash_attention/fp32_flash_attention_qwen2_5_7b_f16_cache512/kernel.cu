/**
 * Flash Attention Kernel for Qwen2.5-7B with FP16 KV Cache (seq_len=512)
 *
 * Parameters:
 *   - seq_len: 512 (KV cache sequence length)
 *   - num_heads: 28 (Qwen2.5-7B has 28 heads)
 *   - head_dim: 128
 *   - KV cache type: FP16
 *   - Query: FP32
 *
 * Formula: attn = softmax(Q @ K^T / sqrt(head_dim)) @ V
 *
 * Baseline (RTX4090):
 *   - batch_1: ~1 TFLOPS
 *   - batch_512: ~130 TFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

// Single-threaded reference kernel for correctness
__global__ void flash_attn_f16_ref_kernel(
    const float* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    float* __restrict__ output,
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    const float scale = 1.0f / sqrtf((float)head_dim);

    const float* q = query + (batch_idx * num_heads + head_idx) * head_dim;
    float* out = output + (batch_idx * num_heads + head_idx) * head_dim;

    // Shared memory for K and V buffers
    extern __shared__ float shared_mem[];
    float* k_buffer = shared_mem;
    float* v_buffer = k_buffer + head_dim;

    // Online softmax state
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;

    // Initialize output
    for (int d = 0; d < head_dim; d++) {
        out[d] = 0.0f;
    }

    // Process all KV positions
    for (int s = 0; s < seq_len; s++) {
        // Load K from FP16 cache
        const half* k = key_cache + (s * num_heads + head_idx) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            k_buffer[d] = __half2float(k[d]);
        }

        // Compute attention score: Q @ K^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * k_buffer[d];
        }
        score *= scale;

        // Online softmax update
        float new_max = fmaxf(max_score, score);
        float exp_old = expf(max_score - new_max);
        float exp_new = expf(score - new_max);

        sum_exp = sum_exp * exp_old + exp_new;

        // Scale existing output
        for (int d = 0; d < head_dim; d++) {
            out[d] *= exp_old;
        }

        // Load V from FP16 cache
        const half* v = value_cache + (s * num_heads + head_idx) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            v_buffer[d] = __half2float(v[d]);
        }

        // Accumulate weighted V
        for (int d = 0; d < head_dim; d++) {
            out[d] += exp_new * v_buffer[d];
        }

        max_score = new_max;
    }

    // Normalize by sum_exp
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int d = 0; d < head_dim; d++) {
        out[d] *= inv_sum;
    }
}

// PyTorch interface
torch::Tensor forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    auto output = torch::empty({batch_size, num_heads, head_dim},
        torch::dtype(torch::kFloat32).device(query.device()));

    // Shared memory: 2 * head_dim floats for K and V buffers
    size_t shared_mem = 2 * head_dim * sizeof(float);

    dim3 grid(batch_size, num_heads);
    dim3 block(1);  // Single thread per (batch, head) pair

    flash_attn_f16_ref_kernel<<<grid, block, shared_mem>>>(
        query.data_ptr<float>(),
        key_cache.data_ptr<at::Half>(),
        value_cache.data_ptr<at::Half>(),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention FP16 KV for Qwen2.5-7B");
}
