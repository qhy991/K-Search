/**
 * Flash Attention Kernel for Qwen2.5-7B with Q4_0 KV Cache
 *
 * Q4_0 Format (18 bytes per 32 values):
 * - 2 bytes: FP16 scale
 * - 16 bytes: packed 4-bit values
 * - Packing: packed[i] = q[i] | (q[16+i] << 4)
 *
 * Unpacking:
 * - For i in [0, 15]: value[i] = (packed[i] & 0x0F - 8) * scale
 * - For i in [16, 31]: value[i] = ((packed[i-16] >> 4) & 0x0F - 8) * scale
 *
 * Configuration:
 * - num_heads: 28
 * - head_dim: 128
 * - seq_len: 8192
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

constexpr int HEAD_DIM = 128;
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 256;
constexpr int Q4_BLOCK_SIZE = 32;
constexpr int Q4_BYTES_PER_BLOCK = 18;
constexpr int NUM_Q4_BLOCKS = HEAD_DIM / Q4_BLOCK_SIZE;  // 4

__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

/**
 * Dequantize Q4_0 block (32 values from 18 bytes)
 */
__device__ void dequantize_q4_block(
    const uint8_t* q4_block,
    float* output
) {
    float scale = __half2float(*reinterpret_cast<const half*>(q4_block));
    const uint8_t* packed = q4_block + 2;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t byte_val = packed[i];
        output[i] = ((byte_val & 0x0F) - 8.0f) * scale;
        output[16 + i] = (((byte_val >> 4) & 0x0F) - 8.0f) * scale;
    }
}

/**
 * Dequantize single value from Q4_0 block
 */
__device__ float dequantize_q4_value(
    const uint8_t* q4_block,
    int dim_in_block
) {
    float scale = __half2float(*reinterpret_cast<const half*>(q4_block));
    const uint8_t* packed = q4_block + 2;
    
    if (dim_in_block < 16) {
        return ((packed[dim_in_block] & 0x0F) - 8.0f) * scale;
    } else {
        return (((packed[dim_in_block - 16] >> 4) & 0x0F) - 8.0f) * scale;
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE) flash_attn_q4_0_kernel(
    const float* __restrict__ query,
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
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

    __shared__ float s_query[HEAD_DIM];
    __shared__ float s_scores[TILE_SIZE];
    __shared__ float s_warp_max[8];
    __shared__ float s_warp_sum[8];
    __shared__ float s_output[HEAD_DIM];

    // Load query
    if (tid < HEAD_DIM) {
        s_query[tid] = __ldg(query + (batch_idx * num_heads + head_idx) * HEAD_DIM + tid);
    }
    __syncthreads();

    float global_max = -FLT_MAX;
    float global_sum = 0.0f;
    float acc = 0.0f;
    const int out_dim = tid;

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, seq_len);
        int tile_len = tile_end - tile_start;

        // Phase 1: Compute attention scores
        for (int s = tid; s < tile_len; s += BLOCK_SIZE) {
            int seq_idx = tile_start + s;
            float score = 0.0f;

            for (int b = 0; b < NUM_Q4_BLOCKS; b++) {
                int base_offset = seq_idx * (num_heads * NUM_Q4_BLOCKS) + head_idx * NUM_Q4_BLOCKS;
                const uint8_t* q4_block = key_cache + (base_offset + b) * Q4_BYTES_PER_BLOCK;

                float dequantized[32];
                dequantize_q4_block(q4_block, dequantized);

                int dim_start = b * Q4_BLOCK_SIZE;
                for (int d = 0; d < Q4_BLOCK_SIZE; d++) {
                    score += s_query[dim_start + d] * dequantized[d];
                }
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
        for (int w = 1; w < num_warps && w < 8; w++) {
            tile_max = fmaxf(tile_max, s_warp_max[w]);
        }

        // Phase 3: Online softmax
        float new_global_max = fmaxf(global_max, tile_max);
        float scale_old = (global_max > -FLT_MAX / 2) ? expf(global_max - new_global_max) : 1.0f;

        if (out_dim < HEAD_DIM) {
            acc *= scale_old;
        }

        float local_sum = 0.0f;
        for (int s = tid; s < tile_len; s += BLOCK_SIZE) {
            s_scores[s] = expf(s_scores[s] - new_global_max);
            local_sum += s_scores[s];
        }
        local_sum = warp_reduce_sum(local_sum);

        if (lane_id == 0) s_warp_sum[warp_id] = local_sum;
        __syncthreads();

        float tile_sum = s_warp_sum[0];
        for (int w = 1; w < num_warps && w < 8; w++) {
            tile_sum += s_warp_sum[w];
        }

        global_sum = global_sum * scale_old + tile_sum;
        global_max = new_global_max;

        // Phase 4: Value accumulation
        if (out_dim < HEAD_DIM) {
            int out_block = out_dim / Q4_BLOCK_SIZE;
            int out_dim_in_block = out_dim % Q4_BLOCK_SIZE;

            for (int s = 0; s < tile_len; s++) {
                int seq_idx = tile_start + s;
                float attn = s_scores[s];

                int base_offset = seq_idx * (num_heads * NUM_Q4_BLOCKS) + head_idx * NUM_Q4_BLOCKS;
                const uint8_t* v_block = value_cache + (base_offset + out_block) * Q4_BYTES_PER_BLOCK;

                float v_val = dequantize_q4_value(v_block, out_dim_in_block);
                acc += attn * v_val;
            }
        }
        __syncthreads();
    }

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

    flash_attn_q4_0_kernel<<<grid, block>>>(
        query.data_ptr<float>(),
        key_cache.data_ptr<uint8_t>(),
        value_cache.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention Q4_0 for Qwen2.5-7B");
}
