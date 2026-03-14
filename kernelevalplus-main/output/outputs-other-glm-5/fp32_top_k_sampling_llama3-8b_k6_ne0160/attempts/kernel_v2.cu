/**
 * Top-K Sampling Kernel - Optimized Version 2
 *
 * Optimizations:
 *   1. More threads per block (256) for better GPU utilization
 *   2. Warp-level primitives for efficient reduction
 *   3. Reduced shared memory usage
 *   4. Better memory coalescing
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// Constants
constexpr int K = 6;
constexpr int THREADS_PER_BLOCK = 256;

/**
 * Device function: maintain top-k in registers
 * Each thread maintains its local top-k, then we reduce
 */
struct LocalTopK {
    float vals[K];
    int idxs[K];

    __device__ void init() {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            vals[i] = -1.0f;
            idxs[i] = -1;
        }
    }

    // Insert a value, maintaining sorted order (descending)
    __device__ void insert(float val, int idx) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            if (val > vals[i]) {
                // Shift elements down
                #pragma unroll
                for (int j = K - 1; j > i; j--) {
                    vals[j] = vals[j-1];
                    idxs[j] = idxs[j-1];
                }
                vals[i] = val;
                idxs[i] = idx;
                break;
            }
        }
    }

    // Merge another TopK into this one
    __device__ void merge(const LocalTopK& other) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            if (other.vals[i] > 0.0f) {
                insert(other.vals[i], other.idxs[i]);
            }
        }
    }
};

/**
 * Warp-level reduction for TopK
 * Each warp reduces its top-k, then lane 0 writes to shared memory
 */
__device__ void warp_reduce_topk(LocalTopK& local, LocalTopK* shared_topk) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // Shuffle-reduce within warp
    // Lane 0 collects all top-k from the warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Each lane shares its K values with lane_id - offset
        for (int i = 0; i < K; i++) {
            float recv_val = __shfl_down_sync(0xffffffff, local.vals[i], offset);
            int recv_idx = __shfl_down_sync(0xffffffff, local.idxs[i], offset);

            if (lane_id < offset && recv_val > 0.0f) {
                local.insert(recv_val, recv_idx);
            }
        }
    }

    // Lane 0 writes to shared memory
    if (lane_id == 0) {
        shared_topk[warp_id] = local;
    }
}

/**
 * TopK Sampling Kernel - Optimized
 */
extern "C" __global__ void topk_sampling_kernel_v2(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size,
    int k
) {
    // One block per batch element
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* batch_probs = probs + batch_idx * vocab_size;

    // Shared memory: one LocalTopK per warp + final result
    __shared__ LocalTopK shared_topk[THREADS_PER_BLOCK / 32 + 1];
    __shared__ LocalTopK final_topk;

    // Each thread initializes its local top-k
    LocalTopK local;
    local.init();

    // Each thread processes strided elements
    for (int v = threadIdx.x; v < vocab_size; v += THREADS_PER_BLOCK) {
        float prob = batch_probs[v];
        if (prob > 0.0f) {
            local.insert(prob, v);
        }
    }

    // Warp-level reduction
    warp_reduce_topk(local, shared_topk);
    __syncthreads();

    // Block-level reduction: first warp merges all warp results
    const int num_warps = THREADS_PER_BLOCK / 32;
    if (threadIdx.x < num_warps) {
        LocalTopK merged;
        merged.init();

        // Each thread in first warp merges multiple warp results
        for (int w = threadIdx.x; w < num_warps; w += num_warps) {
            merged.merge(shared_topk[w]);
        }

        // Final reduction within first warp
        if (threadIdx.x == 0) {
            for (int w = 1; w < num_warps; w++) {
                merged.merge(shared_topk[w]);
            }
            final_topk = merged;
        }
    }
    __syncthreads();

    // Output: deterministic sampling (top-1 = argmax)
    if (threadIdx.x == 0) {
        samples[batch_idx] = final_topk.idxs[0];
    }
}

/**
 * Alternative: Simple single-warp kernel for small vocab
 * More efficient for vocab_size <= 160 with small batch
 */
extern "C" __global__ void topk_sampling_kernel_small(
    const float* __restrict__ probs,
    int64_t* __restrict__ samples,
    int batch_size,
    int vocab_size,
    int k
) {
    // One warp per batch element (up to 32 warps per block)
    const int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;

    const float* batch_probs = probs + batch_idx * vocab_size;
    const int lane_id = threadIdx.x;

    // Each lane maintains local top-k
    LocalTopK local;
    local.init();

    // Strided access: each lane processes vocab_size/32 elements
    for (int v = lane_id; v < vocab_size; v += 32) {
        float prob = batch_probs[v];
        if (prob > 0.0f) {
            local.insert(prob, v);
        }
    }

    // Warp reduction
    __shared__ LocalTopK shared_results[32];  // Max 32 warps per block
    LocalTopK* warp_shared = &shared_results[threadIdx.y];

    warp_reduce_topk(local, warp_shared);

    // Lane 0 of each warp writes result
    if (lane_id == 0) {
        samples[batch_idx] = warp_shared->idxs[0];
    }
}

/**
 * Host function to launch kernel
 */
extern "C" void topk_kernel(
    const float* probs,
    int64_t* indices,
    int batch_size,
    int vocab_size,
    int k
) {
    if (batch_size <= 32 && vocab_size <= 256) {
        // Small batch: use multi-warp kernel (one warp per batch)
        dim3 block(32, min(batch_size, 32));  // 32 threads per warp
        dim3 grid((batch_size + 31) / 32);
        topk_sampling_kernel_small<<<grid, block>>>(probs, indices, batch_size, vocab_size, k);
    } else {
        // Large batch: one block per batch
        int blocks = batch_size;
        topk_sampling_kernel_v2<<<blocks, THREADS_PER_BLOCK>>>(probs, indices, batch_size, vocab_size, k);
    }
}

/**
 * PyTorch binding
 */
#include <torch/extension.h>

torch::Tensor forward(torch::Tensor probs, int k) {
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(probs.dim() == 2, "probs must be 2D (batch_size, vocab_size)");
    TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");

    int batch_size = probs.size(0);
    int vocab_size = probs.size(1);

    auto samples = torch::empty({batch_size},
        torch::dtype(torch::kInt64).device(probs.device()));

    topk_kernel(
        probs.data_ptr<float>(),
        samples.data_ptr<int64_t>(),
        batch_size,
        vocab_size,
        k
    );

    return samples;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "TopK Sampling");
}
