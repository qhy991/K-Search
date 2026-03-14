/**
 * Flash Attention Kernel for Qwen2.5-7B with F16 KV Cache - v2 Optimized
 *
 * Key Optimizations:
 * 1. 512 threads for better parallelism
 * 2. 512-element tiles for better memory throughput
 * 3. Vectorized half2 loads
 * 4. Reduced synchronization
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

constexpr int HEAD_DIM = 128;
constexpr int BLOCK_SIZE = 512;
constexpr int TILE_SIZE = 512;

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;
}

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// Vectorized load for half2
__device__ __forceinline__ void load_half2(const half* ptr, float& v0, float& v1) {
    half2 val = *reinterpret_cast<const half2*>(ptr);
    v0 = __half2float(val.x);
    v1 = __half2float(val.y);
}

__global__ void __launch_bounds__(BLOCK_SIZE) flash_attn_f16_kernel(
    const float* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    float* __restrict__ output,
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = BLOCK_SIZE / 32;

    // Shared memory
    __shared__ float s_query[HEAD_DIM];
    __shared__ float s_scores[TILE_SIZE];
    __shared__ float s_warp_max[16];
    __shared__ float s_warp_sum[16];
    __shared__ float s_output[HEAD_DIM];

    // Load query
    if (tid < HEAD_DIM) {
        s_query[tid] = __ldg(query + (batch_idx * num_heads + head_idx) * HEAD_DIM + tid);
    }
    __syncthreads();

    // Global softmax state
    float global_max = -FLT_MAX;
    float global_sum = 0.0f;
    float acc = 0.0f;
    const int out_dim = tid;

    // Process sequence in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, seq_len);
        int tile_len = tile_end - tile_start;

        // Phase 1: Compute attention scores with vectorized loads
        for (int s = tid; s < tile_len; s += BLOCK_SIZE) {
            int seq_idx = tile_start + s;
            const half* k = key_cache + (seq_idx * num_heads + head_idx) * HEAD_DIM;

            float score = 0.0f;
            
            // Use half2 vectorized loads (64 pairs for 128 dims)
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 2) {
                float k0, k1;
                load_half2(k + d, k0, k1);
                score += s_query[d] * k0 + s_query[d + 1] * k1;
            }

            s_scores[s] = score * scale;
        }
        __syncthreads();

        // Phase 2: Find max
        float local_max = -FLT_MAX;
        for (int s = tid; s < tile_len; s += BLOCK_SIZE) {
            local_max = fmaxf(local_max, s_scores[s]);
        }
        local_max = warp_reduce_max(local_max);

        if (lane_id == 0) s_warp_max[warp_id] = local_max;
        __syncthreads();

        float tile_max = s_warp_max[0];
        for (int w = 1; w < num_warps && w < 16; w++) {
            tile_max = fmaxf(tile_max, s_warp_max[w]);
        }

        // Phase 3: Online softmax
        float new_global_max = fmaxf(global_max, tile_max);
        float scale_old = (global_max > -FLT_MAX / 2) ? expf(global_max - new_global_max) : 1.0f;

        if (out_dim < HEAD_DIM) {
            acc *= scale_old;
        }

        // Compute exp and sum
        float local_sum = 0.0f;
        for (int s = tid; s < tile_len; s += BLOCK_SIZE) {
            s_scores[s] = expf(s_scores[s] - new_global_max);
            local_sum += s_scores[s];
        }
        local_sum = warp_reduce_sum(local_sum);

        if (lane_id == 0) s_warp_sum[warp_id] = local_sum;
        __syncthreads();

        float tile_sum = s_warp_sum[0];
        for (int w = 1; w < num_warps && w < 16; w++) {
            tile_sum += s_warp_sum[w];
        }

        global_sum = global_sum * scale_old + tile_sum;
        global_max = new_global_max;

        // Phase 4: Value accumulation
        if (out_dim < HEAD_DIM) {
            for (int s = 0; s < tile_len; s++) {
                int seq_idx = tile_start + s;
                float attn = s_scores[s];

                const half* v = value_cache + (seq_idx * num_heads + head_idx) * HEAD_DIM;
                float v_val = __half2float(__ldg(v + out_dim));

                acc += attn * v_val;
            }
        }
        __syncthreads();
    }

    // Final normalization
    if (out_dim < HEAD_DIM) {
        s_output[out_dim] = acc / global_sum;
    }
    __syncthreads();

    if (tid < HEAD_DIM) {
        output[(batch_idx * num_heads + head_idx) * HEAD_DIM + tid] = s_output[tid];
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

    dim3 grid(batch_size, num_heads);
    dim3 block(BLOCK_SIZE);

    flash_attn_f16_kernel<<<grid, block>>>(
        query.data_ptr<float>(),
        reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention F16 v2 for Qwen2.5-7B");
}
