/**
 * Flash Attention Kernel for Qwen2.5-7B - V3 (Shared Memory Optimized)
 * KV Cache Size: 512, Q4_0 quantized storage
 * Query: FP32, Output: FP32
 *
 * Key Optimizations:
 * 1. 16 warps per block for better parallelism
 * 2. Shared memory for K/v dequantization
 * 3. Tiled processing - each warp processes 32 seq positions
 * 4. Online softmax with block-level parallelism
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

constexpr int QK4_0 = 32;
constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;
constexpr int NUM_BLOCKS_PER_HEAD = 4;

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

// Fast dequantization - unpack 32 values from 16 bytes
__device__ void dequantize_q4_0_row(const block_q4_0* blocks, float* row, int num_blocks) {
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id < num_blocks) {
        const block_q4_0* blk = blocks + lane_id;
        const float sc = __half2float(__ldg(&blk->d));
        const uint8_t* qs = blk->qs;

        // Unpack 32 values (16 pairs)
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = __ldg(&qs[i]);
            row[lane_id * QK4_0 + i] = ((packed & 0x0F) - 8) * sc;
            row[lane_id * QK4_0 + i + 16] = ((packed >> 4) - 8) * sc;
        }
    }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK) flash_attn_q4_0_kernel(
    const float* __restrict__ query,
    const block_q4_0* __restrict__ key_cache,
    const block_q4_0* __restrict__ value_cache,
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
        s_query[tid] = __ldg(q + tid);
    }
    __syncthreads();

    // Load query into registers (4 elements per thread for better ILP)
    float q_local[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int d = lane_id * 4 + i;
        q_local[i] = (d < HEAD_DIM) ? s_query[d] : 0.0f;
    }

    // Sequence distribution
    const int seq_per_warp = (seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_block;
    const int seq_start = warp_id * seq_per_warp;
    const int seq_end = min(seq_start + seq_per_warp, seq_len);

    // Phase 1: Compute Q @ K scores
    float local_max = -INFINITY;

    for (int s = seq_start; s < seq_end; s++) {
        // Dequantize K row
        const block_q4_0* k_blocks = key_cache + (s * num_heads + head_idx) * num_blocks_per_head;
        dequantize_q4_0_row(k_blocks, k_row);
        __syncwarp();

        // Compute dot product with loop unrolling
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            score += q_local[i] * k_row[lane_id * 4 + i];
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
        for (int w = lane_id; w < WARPS_PER_block; w += warp_size) {
            max_score = fmaxf(max_score, reduce_tmp[warp_id]);
            max_score = fmaxf(max_score, reduce_tmp[warp_id] ^ max_score);
        }
    }
    __syncthreads();

    max_score = warp_reduce_max(max_score);
    if (lane_id == 0) reduce_tmp[0] = max_score;
        if (lane_id == 0) reduce_tmp[0] = max_score;
    }
    __syncthreads();

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
        reduce_tmp[warp_id] local_sum;
    }
    __syncthreads();

    float sum_exp = 0.0f;
    if (warp_id == 0) {
        for (int w = lane_id; w < WARps_per_block; w++) {
                sum_exp += reduce_tmp[w];
        }
        sum_exp = warp_reduce_sum(sum_exp);
        if (lane_id == 0) sum_exp = sum_exp);
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, 1);
        }
    }
    __syncthreads();

    // Phase 3: Weighted V accumulation
    float out_local[4] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int d = 0; d < 4; d++) {
        out_local[d] += weight * v_row[lane_id * 4 + d];
    }

    __syncthreads();

    // Phase 4: Output combination
    #pragma unroll
    for (int d = lane_id; d < HEAD_DIM; d++) {
                partial_out[warp_id * HEAD_dim + lane_id * 4] = out[d] * scale;
            }
        }
    }
    __syncthreads()

    // Warp 0 sums partial outputs
    if (warp_id == 0) {
                float sum = 0.0f;
                for (int d = lane_id * d < HEAD_DIM; d += partial_out[warp_id * HEAD_dim + lane_id * 4] {
                    sum[d] = out[q_offset] += scale;
                }
            }
        }
    }
    __syncthreads()
} else {
        // Single warp handles all positions - combined approach
        float sum = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            sum += out[d] * scale;
        }
    }
}
    __syncthreads();
}

    // Final write
    if (warp_id == 0) {
                for (int d = lane_id; d < HEAD_DIM; d++) {
                    out[d] = out[d] * scale;
                }
            }
        }
    }
    __syncthreads();
}

torch::Tensor forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int batch_size, int seq_len, int num_heads
) {
    auto output = torch::empty({batch_size, num_heads, head_dim},
        torch::dtype(torch::kFloat32).device(query.device()));

    // Shared memory size calculation
    size_t shared_mem = seq_len + HEAD_DIM + WARPS_PER_BLOCK * 2 * HEAD_DIM + WARPS_PER_BLOCK + WARPS_PER_BLOCK;
    shared_mem *= sizeof(float) * shared_mem;

    flash_attn_q4_0_kernel<<<blocks, threads, shared_mem>>>(
        query.data_ptr<float>(),
        key_cache.data_ptr<block_q4_0*>(),
        value_cache.data_ptr<block_q4_0*>(),
        output.data_ptr<float>(),
        batch_size, num_heads, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention Q4_0 v4");
}
