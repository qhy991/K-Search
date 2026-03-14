/**
 * TopK Sampling CUDA Kernel - LLaMA-3-8B Full Vocabulary (k=8) - Optimized v2
 *
 * Target: LLaMA-3-8B with k=8, vocab_size=128256
 * Hardware: NVIDIA RTX 4090 (Compute 8.9, 128 SMs, 6MB L2 Cache, 1008 GB/s BW)
 *
 * Key Optimization: Use "min-of-topk" approach instead of sorted insertion
 *   - Track only the minimum value in top-k
 *   - Replace min when finding larger value
 *   - Much fewer comparisons: O(1) amortized vs O(k) per element
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

constexpr int WARP_SIZE = 32;
constexpr int K = 8;

/**
 * Find minimum value index in top-k array
 */
__device__ __forceinline__ int find_min_idx(float* vals, int count) {
    int min_idx = 0;
    float min_val = vals[0];
    for (int i = 1; i < count; i++) {
        if (vals[i] < min_val) {
            min_val = vals[i];
            min_idx = i;
        }
    }
    return min_idx;
}

/**
 * Optimized top-k update: only update if value > min_of_topk
 */
__device__ __forceinline__ void update_topk_optimized(
    float val, int idx,
    float* top_vals, int* top_indices, int* count
) {
    if (*count < K) {
        // Still filling top-k
        top_vals[*count] = val;
        top_indices[*count] = idx;
        (*count)++;
    } else {
        // Find minimum in current top-k
        int min_idx = find_min_idx(top_vals, K);

        // Replace if new value is larger
        if (val > top_vals[min_idx] ||
            (val == top_vals[min_idx] && idx < top_indices[min_idx])) {
            top_vals[min_idx] = val;
            top_indices[min_idx] = idx;
        }
    }
}

/**
 * Sort top-k by value (descending) - only done once at the end
 */
__device__ __forceinline__ void sort_topk(float* vals, int* indices) {
    // Simple bubble sort for k=8
    #pragma unroll
    for (int i = 0; i < K - 1; i++) {
        #pragma unroll
        for (int j = 0; j < K - i - 1; j++) {
            if (vals[j] < vals[j + 1] ||
                (vals[j] == vals[j + 1] && indices[j] > indices[j + 1])) {
                // Swap
                float tmp_val = vals[j];
                vals[j] = vals[j + 1];
                vals[j + 1] = tmp_val;

                int tmp_idx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tmp_idx;
            }
        }
    }
}

/**
 * Merge two top-k arrays - optimized version
 */
__device__ __forceinline__ void merge_topk_opt(
    float* vals_a, int* idx_a,
    float* vals_b, int* idx_b,
    float* out_vals, int* out_idx
) {
    int a_pos = 0, b_pos = 0;

    #pragma unroll
    for (int i = 0; i < K; i++) {
        bool take_a = false;

        if (a_pos < K && b_pos < K) {
            take_a = (vals_a[a_pos] > vals_b[b_pos]) ||
                     (vals_a[a_pos] == vals_b[b_pos] && idx_a[a_pos] < idx_b[b_pos]);
        } else if (a_pos < K) {
            take_a = true;
        }

        if (take_a) {
            out_vals[i] = vals_a[a_pos];
            out_idx[i] = idx_a[a_pos];
            a_pos++;
        } else if (b_pos < K) {
            out_vals[i] = vals_b[b_pos];
            out_idx[i] = idx_b[b_pos];
            b_pos++;
        } else {
            out_vals[i] = -FLT_MAX;
            out_idx[i] = -1;
        }
    }
}

/**
 * Kernel A: One warp per batch element with vectorized loads
 * Optimized with min-heap style top-k tracking
 */
__global__ void topk_warp_per_batch_k8_v2(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= batch_size) return;

    const float* prob = probs + warp_id * vocab_size;

    // Each thread maintains local top-k
    float local_vals[K];
    int local_indices[K];
    int local_count = 0;

    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_vals[i] = -FLT_MAX;
        local_indices[i] = -1;
    }

    // Vectorized loads
    const float4* prob_vec = reinterpret_cast<const float4*>(prob);
    const int num_vec = vocab_size >> 2;

    for (int v = lane_id; v < num_vec; v += WARP_SIZE) {
        float4 vec = prob_vec[v];
        int base = v << 2;

        update_topk_optimized(vec.x, base, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.y, base + 1, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.z, base + 2, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.w, base + 3, local_vals, local_indices, &local_count);
    }

    // Handle remainder
    for (int v = (num_vec << 2) + lane_id; v < vocab_size; v += WARP_SIZE) {
        float val = prob[v];
        update_topk_optimized(val, v, local_vals, local_indices, &local_count);
    }

    // Sort local top-k for merging
    sort_topk(local_vals, local_indices);

    // Warp reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        float partner_vals[K];
        int partner_indices[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            partner_vals[i] = __shfl_down_sync(0xffffffff, local_vals[i], offset);
            partner_indices[i] = __shfl_down_sync(0xffffffff, local_indices[i], offset);
        }

        if (lane_id < offset) {
            float merged_vals[K];
            int merged_indices[K];
            merge_topk_opt(
                local_vals, local_indices,
                partner_vals, partner_indices,
                merged_vals, merged_indices
            );
            #pragma unroll
            for (int i = 0; i < K; i++) {
                local_vals[i] = merged_vals[i];
                local_indices[i] = merged_indices[i];
            }
        }
    }

    if (lane_id == 0) {
        samples[warp_id] = local_indices[0];
    }
}

/**
 * Kernel B: Multiple batches per warp (4 batches per warp)
 */
__global__ void topk_multi_batch_per_warp_k8_v2(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int batch_in_warp = lane_id >> 3;
    int sub_lane = lane_id & 7;

    int batch_idx = (warp_id << 2) + batch_in_warp;
    if (batch_idx >= batch_size) return;

    const float* prob = probs + batch_idx * vocab_size;

    float local_vals[K];
    int local_indices[K];
    int local_count = 0;

    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_vals[i] = -FLT_MAX;
        local_indices[i] = -1;
    }

    const float4* prob_vec = reinterpret_cast<const float4*>(prob);
    const int num_vec = vocab_size >> 2;

    for (int v = sub_lane; v < num_vec; v += 8) {
        float4 vec = prob_vec[v];
        int base = v << 2;

        update_topk_optimized(vec.x, base, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.y, base + 1, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.z, base + 2, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.w, base + 3, local_vals, local_indices, &local_count);
    }

    for (int v = (num_vec << 2) + sub_lane; v < vocab_size; v += 8) {
        float val = prob[v];
        update_topk_optimized(val, v, local_vals, local_indices, &local_count);
    }

    sort_topk(local_vals, local_indices);

    // Reduction within 8-thread group
    for (int offset = 4; offset > 0; offset >>= 1) {
        float partner_vals[K];
        int partner_indices[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            partner_vals[i] = __shfl_down_sync(0xffffffff, local_vals[i], offset);
            partner_indices[i] = __shfl_down_sync(0xffffffff, local_indices[i], offset);
        }

        if (sub_lane < offset) {
            float merged_vals[K];
            int merged_indices[K];
            merge_topk_opt(
                local_vals, local_indices,
                partner_vals, partner_indices,
                merged_vals, merged_indices
            );
            #pragma unroll
            for (int i = 0; i < K; i++) {
                local_vals[i] = merged_vals[i];
                local_indices[i] = merged_indices[i];
            }
        }
    }

    if (sub_lane == 0) {
        samples[batch_idx] = local_indices[0];
    }
}

/**
 * Kernel C: One block per batch element with 512 threads for better parallelism
 */
__global__ void topk_block_per_batch_k8_v2(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* prob = probs + batch_idx * vocab_size;

    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    float local_vals[K];
    int local_indices[K];
    int local_count = 0;

    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_vals[i] = -FLT_MAX;
        local_indices[i] = -1;
    }

    const float4* prob_vec = reinterpret_cast<const float4*>(prob);
    const int num_vec = vocab_size >> 2;

    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        float4 vec = prob_vec[v];
        int base = v << 2;

        update_topk_optimized(vec.x, base, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.y, base + 1, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.z, base + 2, local_vals, local_indices, &local_count);
        update_topk_optimized(vec.w, base + 3, local_vals, local_indices, &local_count);
    }

    for (int v = (num_vec << 2) + threadIdx.x; v < vocab_size; v += blockDim.x) {
        float val = prob[v];
        update_topk_optimized(val, v, local_vals, local_indices, &local_count);
    }

    sort_topk(local_vals, local_indices);

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float partner_vals[K];
        int partner_indices[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            partner_vals[i] = __shfl_down_sync(0xffffffff, local_vals[i], offset);
            partner_indices[i] = __shfl_down_sync(0xffffffff, local_indices[i], offset);
        }

        if (lane_id < offset) {
            float merged_vals[K];
            int merged_indices[K];
            merge_topk_opt(
                local_vals, local_indices,
                partner_vals, partner_indices,
                merged_vals, merged_indices
            );
            #pragma unroll
            for (int i = 0; i < K; i++) {
                local_vals[i] = merged_vals[i];
                local_indices[i] = merged_indices[i];
            }
        }
    }

    // Shared memory for warp results
    __shared__ float s_warp_vals[16][K];  // Max 16 warps (512 threads)
    __shared__ int s_warp_indices[16][K];

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            s_warp_vals[warp_id][i] = local_vals[i];
            s_warp_indices[warp_id][i] = local_indices[i];
        }
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            local_vals[i] = (lane_id < num_warps) ? s_warp_vals[lane_id][i] : -FLT_MAX;
            local_indices[i] = (lane_id < num_warps) ? s_warp_indices[lane_id][i] : -1;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            float partner_vals[K];
            int partner_indices[K];

            #pragma unroll
            for (int i = 0; i < K; i++) {
                partner_vals[i] = __shfl_down_sync(0xffffffff, local_vals[i], offset);
                partner_indices[i] = __shfl_down_sync(0xffffffff, local_indices[i], offset);
            }

            if (lane_id < offset && lane_id + offset < num_warps) {
                float merged_vals[K];
                int merged_indices[K];
                merge_topk_opt(
                    local_vals, local_indices,
                    partner_vals, partner_indices,
                    merged_vals, merged_indices
                );
                #pragma unroll
                for (int i = 0; i < K; i++) {
                    local_vals[i] = merged_vals[i];
                    local_indices[i] = merged_indices[i];
                }
            }
        }

        if (lane_id == 0) {
            samples[batch_idx] = local_indices[0];
        }
    }
}

/**
 * PyTorch Interface with Adaptive Dispatch
 */
torch::Tensor forward(torch::Tensor probs, int k) {
    int batch_size = probs.size(0);
    int vocab_size = probs.size(1);

    TORCH_CHECK(probs.dim() == 2, "probs must be 2D tensor");
    TORCH_CHECK(probs.dtype() == torch::kFloat32, "probs must be float32");
    TORCH_CHECK(probs.is_cuda(), "probs must be CUDA tensor");

    auto samples = torch::empty({batch_size},
        torch::dtype(torch::kInt64).device(probs.device()));

    if (batch_size <= 8) {
        int threads = 256;
        int warps_per_block = threads / WARP_SIZE;
        int blocks = (batch_size + warps_per_block - 1) / warps_per_block;
        topk_warp_per_batch_k8_v2<<<blocks, threads>>>(
            probs.data_ptr<float>(),
            samples.data_ptr<int64_t>(),
            batch_size,
            vocab_size
        );
    } else if (batch_size <= 128) {
        int threads = 256;
        int warps_per_block = threads / WARP_SIZE;
        int batches_per_block = warps_per_block << 2;
        int blocks = (batch_size + batches_per_block - 1) / batches_per_block;
        topk_multi_batch_per_warp_k8_v2<<<blocks, threads>>>(
            probs.data_ptr<float>(),
            samples.data_ptr<int64_t>(),
            batch_size,
            vocab_size
        );
    } else {
        // Use 512 threads for better parallelism with large vocab
        int block_threads = 512;
        topk_block_per_batch_k8_v2<<<batch_size, block_threads>>>(
            probs.data_ptr<float>(),
            samples.data_ptr<int64_t>(),
            batch_size,
            vocab_size
        );
    }

    return samples;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "TopK Sampling k=8 for LLaMA-3-8B v2");
}
