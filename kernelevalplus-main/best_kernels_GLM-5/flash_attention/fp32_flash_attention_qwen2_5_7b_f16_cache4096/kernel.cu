/**
 * Flash Attention Kernel for Qwen2.5-7B with F16 KV Cache - Optimized V2
 *
 * Key Optimizations:
 * 1. 8 warps per block (256 threads) for good occupancy
 * 2. Online softmax with correct cross-warp reduction
 * 3. Vectorized F16 loads
 * 4. Minimized shared memory usage
 *
 * Parameters:
 * - seq_len: 4096
 * - num_heads: 28
 * - head_dim: 128
 *
 * Performance Target:
 * - RTX4090 baseline: 144.02 TFLOPS (batch=512)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;  // 256
constexpr int HEAD_DIM = 128;

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

__global__ void flash_attn_f16_v2_kernel(
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

    // Shared memory
    __shared__ float smem_max[WARPS_PER_BLOCK];
    __shared__ float smem_sum[WARPS_PER_BLOCK];
    __shared__ float smem_out[WARPS_PER_BLOCK][HEAD_DIM];

    // Load query into registers
    float q_local[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int d = lane_id * 4 + i;
        q_local[i] = (d < HEAD_DIM) ? q[d] : 0.0f;
    }

    // Sequence distribution
    const int seq_per_warp = (seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int seq_start = warp_id * seq_per_warp;
    const int seq_end = min(seq_start + seq_per_warp, seq_len);

    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Process sequence positions
    for (int s = seq_start; s < seq_end; s++) {
        // Load K row
        const half* k_ptr = key_cache + (s * num_heads + head_idx) * HEAD_DIM;
        float k_local[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = lane_id * 4 + i;
            k_local[i] = (d < HEAD_DIM) ? __half2float(k_ptr[d]) : 0.0f;
        }

        // Compute score
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            score += q_local[i] * k_local[i];
        }
        score = warp_reduce_sum(score) * scale;

        // Broadcast from lane 0
        score = __shfl_sync(0xffffffff, score, 0);

        // Online softmax update
        float new_max = fmaxf(max_score, score);
        float scale_factor = expf(max_score - new_max);
        float exp_score = expf(score - new_max);

        sum_exp = sum_exp * scale_factor + exp_score;

        // Load V row and accumulate
        const half* v_ptr = value_cache + (s * num_heads + head_idx) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = lane_id * 4 + i;
            if (d < HEAD_DIM) {
                float v_val = __half2float(v_ptr[d]);
                out_local[i] = out_local[i] * scale_factor + exp_score * v_val;
            }
        }

        max_score = new_max;
    }

    // Cross-warp reduction
    __syncthreads();

    // Store local max
    if (lane_id == 0) {
        smem_max[warp_id] = max_score;
    }
    __syncthreads();

    // Compute global max (warp 0 does this)
    float global_max = -INFINITY;
    if (warp_id == 0) {
        float m = (lane_id < WARPS_PER_BLOCK) ? smem_max[lane_id] : -INFINITY;
        global_max = warp_reduce_max(m);
        if (lane_id == 0) smem_max[0] = global_max;
    }
    __syncthreads();
    global_max = smem_max[0];

    // Scale sum_exp by exp(local_max - global_max)
    float scaled_sum = sum_exp * expf(max_score - global_max);

    // Store scaled sum
    if (lane_id == 0) {
        smem_sum[warp_id] = scaled_sum;
    }
    __syncthreads();

    // Compute global sum (warp 0)
    float global_sum = 0.0f;
    if (warp_id == 0) {
        float s = (lane_id < WARPS_PER_BLOCK) ? smem_sum[lane_id] : 0.0f;
        global_sum = warp_reduce_sum(s);
        if (lane_id == 0) smem_sum[0] = global_sum;
    }
    __syncthreads();
    global_sum = smem_sum[0];

    // Compute final scale factor
    float final_scale = expf(max_score - global_max) / global_sum;

    // Store scaled partial output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int d = lane_id * 4 + i;
        if (d < HEAD_DIM) {
            smem_out[warp_id][d] = out_local[i] * final_scale;
        }
    }
    __syncthreads();

    // Sum partial outputs (warp 0)
    if (warp_id == 0) {
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                sum += smem_out[w][d];
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

    size_t shared_mem = (WARPS_PER_BLOCK * 2 + WARPS_PER_BLOCK * HEAD_DIM) * sizeof(float);

    const int blocks = batch_size * num_heads;
    const int threads = THREADS_PER_BLOCK;

    flash_attn_f16_v2_kernel<<<blocks, threads, shared_mem>>>(
        query.data_ptr<float>(),
        reinterpret_cast<const half*>(key_cache.data_ptr()),
        reinterpret_cast<const half*>(value_cache.data_ptr()),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention F16 V2");
}
