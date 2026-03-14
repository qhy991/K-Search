/**
 * Flash Attention Kernel for Llama3-8B with Q4_0 KV Cache (seq_len=4096)
 * 16 Warps with Correct Shared Memory Layout
 *
 * Q4_0 Format:
 * - Block size: 32 elements
 * - Block storage: 18 bytes (2 bytes FP16 scale + 16 bytes packed 4-bit values)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;
constexpr int NUM_BLOCKS_PER_HEAD = 4;
constexpr int QK4_0 = 32;

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
    float* attn_weights = shared_mem;                    // [seq_len]
    float* s_query = attn_weights + seq_len;             // [HEAD_DIM]
    float* reduce_tmp = s_query + HEAD_DIM;              // [WARPS_PER_BLOCK]
    float* partial_out = reduce_tmp + WARPS_PER_BLOCK;   // [WARPS_PER_BLOCK * HEAD_DIM]
    float* s_kv = partial_out + WARPS_PER_BLOCK * HEAD_DIM;  // [WARPS_PER_BLOCK * 2 * HEAD_DIM] for K and V

    // Each warp has its own K/V storage
    float* k_row = s_kv + warp_id * 2 * HEAD_DIM;
    float* v_row = k_row + HEAD_DIM;

    // Load query into shared memory
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        s_query[d] = __ldg(q + d);
    }
    __syncthreads();

    // Load query into registers
    float q_local[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int d = lane_id * 4 + i;
        q_local[i] = (d < HEAD_DIM) ? s_query[d] : 0.0f;
    }

    // Distribute sequence across warps
    const int seq_per_warp = (seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int seq_start = warp_id * seq_per_warp;
    const int seq_end = min(seq_start + seq_per_warp, seq_len);

    // Phase 1: Compute Q @ K^T
    float local_max = -INFINITY;

    for (int s = seq_start; s < seq_end; s++) {
        // Dequantize K row
        if (lane_id < NUM_BLOCKS_PER_HEAD) {
            const block_q4_0* blk = key_cache +
                (s * num_heads + head_idx) * NUM_BLOCKS_PER_HEAD + lane_id;
            const float sc = __half2float(__ldg(&blk->d));
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = __ldg(&blk->qs[i]);
                k_row[lane_id * QK4_0 + i] = ((packed & 0x0F) - 8) * sc;
                k_row[lane_id * QK4_0 + i + 16] = ((packed >> 4) - 8) * sc;
            }
        }
        __syncwarp();

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
        // Dequantize V row
        if (lane_id < NUM_BLOCKS_PER_HEAD) {
            const block_q4_0* blk = value_cache +
                (s * num_heads + head_idx) * NUM_BLOCKS_PER_HEAD + lane_id;
            const float sc = __half2float(__ldg(&blk->d));
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t packed = __ldg(&blk->qs[i]);
                v_row[lane_id * QK4_0 + i] = ((packed & 0x0F) - 8) * sc;
                v_row[lane_id * QK4_0 + i + 16] = ((packed >> 4) - 8) * sc;
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

    // Store partial output
    int out_base = lane_id * 4;
    if (out_base < HEAD_DIM) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            partial_out[warp_id * HEAD_DIM + out_base + i] = out_local[i];
        }
    }
    __syncthreads();

    // Reduce partial outputs
    if (warp_id == 0) {
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float sum = 0.0f;
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

    // Shared memory: attn_weights + s_query + reduce_tmp + partial_out + s_kv
    size_t shared_mem = seq_len + HEAD_DIM + WARPS_PER_BLOCK + WARPS_PER_BLOCK * HEAD_DIM + WARPS_PER_BLOCK * 2 * HEAD_DIM;
    shared_mem *= sizeof(float);

    const int blocks = batch_size * num_heads;
    const int threads = THREADS_PER_BLOCK;

    flash_attn_q4_0_kernel<<<blocks, threads, shared_mem>>>(
        query.data_ptr<float>(),
        reinterpret_cast<const block_q4_0*>(key_cache.data_ptr()),
        reinterpret_cast<const block_q4_0*>(value_cache.data_ptr()),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention Q4_0");
}
