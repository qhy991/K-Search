/**
 * Flash Attention Kernel for Qwen2.5-7B with F16 KV Cache (seq_len=4096)
 * Multi-Warp Optimized with 16 Warps
 *
 * Key Differences from Q4_0/Q8_0:
 * - KV cache is float16, no quantization/dequantization needed
 * - Just simple __half2float() conversion
 * - num_heads = 28 (vs 32 for Llama3-8B)
 *
 * Performance Target (RTX4090):
 * - batch_1: 3.0 TFLOPS
 * - batch_8: 11.29 TFLOPS
 * - batch_512: 119.48 TFLOPS
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;

// Qwen2.5-7B has 28 attention heads (vs 32 for Llama3-8B)
constexpr int NUM_HEADS = 28;

constexpr int WARPS_PER_BLOCK = 16;
constexpr int THREADS_PER_BLOCK = 512;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK) flash_attn_f16_kernel(
    const float* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    float* __restrict__ output,
    int batch_size, int seq_len, int num_heads
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int head_pair_id = blockIdx.x;

    if (head_pair_id >= batch_size * num_heads) return;

    const int head_idx = head_pair_id % num_heads;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);

    const float* q = query + head_pair_id * HEAD_DIM;
    float* out = output + head_pair_id * HEAD_DIM;

    // Shared memory layout
    extern __shared__ float shared_mem[];
    float* attn_weights = shared_mem;                              // [seq_len]
    float* s_query = attn_weights + seq_len;                       // [HEAD_DIM]
    float* kv_rows = s_query + HEAD_DIM;                           // [WARPS_PER_BLOCK * 2 * HEAD_DIM]
    float* reduce_tmp = kv_rows + WARPS_PER_BLOCK * 2 * HEAD_DIM;  // [WARPS_PER_BLOCK]
    float* partial_out = reduce_tmp + WARPS_PER_BLOCK;             // [WARPS_PER_BLOCK * HEAD_DIM]

    // Each warp's K/V row
    float* k_row = kv_rows + warp_id * 2 * HEAD_DIM;
    float* v_row = k_row + HEAD_DIM;

    // Load query into shared memory
    if (tid < HEAD_DIM) {
        s_query[tid] = q[tid];
    }
    __syncthreads();

    // Load query into registers (4 elements per thread)
    float q_local[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int d = lane_id * 4 + i;
        q_local[i] = (d < HEAD_DIM) ? s_query[d] : 0.0f;
    }

    // Sequence distribution
    const int seq_per_warp = (seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int seq_start = warp_id * seq_per_warp;
    const int seq_end = min(seq_start + seq_per_warp, seq_len);

    // Phase 1: Compute Q @ K scores
    float local_max = -INFINITY;

    for (int s = seq_start; s < seq_end; s++) {
        // Load K row - F16, just convert to float
        const half* k_ptr = key_cache + (s * num_heads + head_idx) * HEAD_DIM;

        // Cooperative load: each thread loads 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = lane_id * 4 + i;
            if (d < HEAD_DIM) {
                k_row[d] = __half2float(__ldg(k_ptr + d));
            }
        }
        __syncwarp();

        // Compute dot product
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = lane_id * 4 + i;
            if (d < HEAD_DIM) {
                score += q_local[i] * k_row[d];
            }
        }
        score = warp_reduce_sum(score) * scale;

        if (lane_id == 0) {
            attn_weights[s] = score;
        }
        local_max = fmaxf(local_max, score);
    }

    // Global max reduction
    __syncthreads();
    if (lane_id == 0) {
        reduce_tmp[warp_id] = local_max;
    }
    __syncthreads();

    float max_score = -INFINITY;
    if (warp_id == 0) {
        for (int w = lane_id; w < WARPS_PER_BLOCK; w += WARP_SIZE) {
            max_score = fmaxf(max_score, reduce_tmp[w]);
        }
        max_score = warp_reduce_max(max_score);
        if (lane_id == 0) reduce_tmp[0] = max_score;
    }
    __syncthreads();
    max_score = reduce_tmp[0];

    // Phase 2: Softmax
    float local_sum = 0.0f;
    for (int s = seq_start; s < seq_end; s++) {
        float exp_val = expf(attn_weights[s] - max_score);
        attn_weights[s] = exp_val;
        local_sum += exp_val;
    }

    // Global sum reduction
    __syncthreads();
    if (lane_id == 0) {
        reduce_tmp[warp_id] = local_sum;
    }
    __syncthreads();

    float sum_exp = 0.0f;
    if (warp_id == 0) {
        for (int w = lane_id; w < WARPS_PER_BLOCK; w += WARP_SIZE) {
            sum_exp += reduce_tmp[w];
        }
        sum_exp = warp_reduce_sum(sum_exp);
        if (lane_id == 0) reduce_tmp[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = reduce_tmp[0];
    const float inv_sum = 1.0f / (sum_exp + 1e-10f);

    // Phase 3: Weighted V accumulation
    float out_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = seq_start; s < seq_end; s++) {
        // Load V row - F16, just convert to float
        const half* v_ptr = value_cache + (s * num_heads + head_idx) * HEAD_DIM;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = lane_id * 4 + i;
            if (d < HEAD_DIM) {
                v_row[d] = __half2float(__ldg(v_ptr + d));
            }
        }
        __syncwarp();

        float weight = attn_weights[s] * inv_sum;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = lane_id * 4 + i;
            if (d < HEAD_DIM) {
                out_local[i] += weight * v_row[d];
            }
        }
    }

    // Phase 4: Output combination across warps
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int d = lane_id * 4 + i;
        if (d < HEAD_DIM) {
            partial_out[warp_id * HEAD_DIM + d] = out_local[i];
        }
    }
    __syncthreads();

    // Warp 0 sums partial outputs
    if (warp_id == 0) {
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                sum += partial_out[w * HEAD_DIM + d];
            }
            out[d] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    auto output = torch::empty({batch_size, num_heads, head_dim},
        torch::dtype(torch::kFloat32).device(query.device()));

    // Shared memory: attn_weights + s_query + kv_rows + reduce_tmp + partial_out
    size_t shared_mem = seq_len + HEAD_DIM + WARPS_PER_BLOCK * 2 * HEAD_DIM + WARPS_PER_BLOCK + WARPS_PER_BLOCK * HEAD_DIM;
    shared_mem *= sizeof(float);

    const int blocks = batch_size * num_heads;
    const int threads = THREADS_PER_BLOCK;

    flash_attn_f16_kernel<<<blocks, threads, shared_mem>>>(
        query.data_ptr<float>(),
        reinterpret_cast<const half*>(key_cache.data_ptr()),
        reinterpret_cast<const half*>(value_cache.data_ptr()),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention F16");
}
