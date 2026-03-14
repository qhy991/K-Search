/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel - Optimized V4
 *
 * Mixtral-8x7B MoE Up projection: N=14336, K=4096
 *
 * Key Optimizations:
 * 1. Warp-level parallelism: 32 threads cooperate on one output
 * 2. Shared memory for activation broadcast
 * 3. Vectorized weight and activation loading
 * 4. Efficient reduction using warp shuffle
 * 5. Multiple outputs per block for better occupancy
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// Q4_0 block structure: 18 bytes
typedef struct {
    uint16_t d;        // FP16 scale
    uint8_t qs[16];    // Packed 4-bit values (32 values total)
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// FP16 to FP32 conversion helper
__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Warp reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// M=1 optimized: Warp per output, process K in parallel
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_fp32_m1_optimized(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K
) {
    const int num_blocks_k = K / 32;
    
    // Each warp (32 threads) computes one output
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    
    if (warp_id >= N) return;
    
    const block_q4_0* w_row = (const block_q4_0*)weight_q + warp_id * num_blocks_k;
    
    float sum = 0.0f;
    
    // Each thread processes num_blocks_k / 32 blocks (if K=4096, 128 blocks / 32 = 4 blocks per thread)
    // For K=4096: 128 blocks, 32 threads -> 4 blocks each
    const int blocks_per_thread = (num_blocks_k + 31) / 32;
    
    for (int b = 0; b < blocks_per_thread; b++) {
        int kb = lane_id * blocks_per_thread + b;
        if (kb >= num_blocks_k) break;
        
        const block_q4_0* w_block = &w_row[kb];
        const float scale = half_to_float(w_block->d);
        const float* act_ptr = activation + kb * 32;
        
        // Process all 32 values in this K-block
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_block->qs[i];
            float w0 = (float)((int)(packed & 0x0F) - 8);
            float w1 = (float)((int)(packed >> 4) - 8);
            sum += scale * w0 * act_ptr[i];
            sum += scale * w1 * act_ptr[i + 16];
        }
    }
    
    // Warp reduce
    sum = warp_reduce_sum(sum);
    
    // Lane 0 writes result
    if (lane_id == 0) {
        output[warp_id] = sum;
    }
}

// ============================================================================
// Small batch kernel (M <= 8): Shared memory + vectorized loads
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q4_0_fp32_small_batch(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    
    // 128 outputs per block
    const int n_base = blockIdx.x * 128;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (n_base >= N || m >= M) return;
    
    // Shared memory for activation (16KB)
    __shared__ float act_shared[4096];
    
    // Cooperative load using vectorized loads
    const float4* act_vec = reinterpret_cast<const float4*>(activation + m * K);
    float4* act_shared_vec = reinterpret_cast<float4*>(act_shared);
    
    // 128 threads, 4096/4 = 1024 float4 to load
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        if (idx < K / 4) {
            act_shared_vec[idx] = act_vec[idx];
        }
    }
    __syncthreads();
    
    // Each thread computes 1 output
    const int n = n_base + tid;
    if (n >= N) return;
    
    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    
    float sum = 0.0f;
    
    // Process K blocks
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0* w_block = &w_row[kb];
        const float scale = half_to_float(w_block->d);
        const float* act_ptr = act_shared + kb * 32;
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_block->qs[i];
            float w0 = (float)((int)(packed & 0x0F) - 8);
            float w1 = (float)((int)(packed >> 4) - 8);
            sum += scale * w0 * act_ptr[i];
            sum += scale * w1 * act_ptr[i + 16];
        }
    }
    
    output[m * N + n] = sum;
}

// ============================================================================
// Medium batch kernel (8 < M <= 64)
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_fp32_medium_batch(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    
    if (n >= N || m >= M) return;
    
    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    const float* act_row = activation + m * K;
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0 w_block = w_row[kb];
        const float scale = half_to_float(w_block.d);
        const float* act_ptr = act_row + kb * 32;
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_block.qs[i];
            float w0 = (float)((int)(packed & 0x0F) - 8);
            float w1 = (float)((int)(packed >> 4) - 8);
            sum += scale * w0 * act_ptr[i];
            sum += scale * w1 * act_ptr[i + 16];
        }
    }
    
    output[m * N + n] = sum;
}

// ============================================================================
// Large batch kernel (M > 64)
// ============================================================================
__global__ void __launch_bounds__(256) gemm_q4_0_fp32_large_batch(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    
    if (n >= N || m >= M) return;
    
    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    const float* act_row = activation + m * K;
    
    float sum = 0.0f;
    
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0 w_block = w_row[kb];
        const float scale = half_to_float(w_block.d);
        const float* act_ptr = act_row + kb * 32;
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w_block.qs[i];
            float w0 = (float)((int)(packed & 0x0F) - 8);
            float w1 = (float)((int)(packed >> 4) - 8);
            sum += scale * w0 * act_ptr[i];
            sum += scale * w1 * act_ptr[i + 16];
        }
    }
    
    output[m * N + n] = sum;
}

// ============================================================================
// PyTorch binding
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));
    
    if (M == 1) {
        // Specialized kernel for M=1: warp per output
        // 256 threads = 8 warps per block, each warp computes 1 output
        // Need ceil(N/8) blocks
        dim3 block(256);
        dim3 grid((N + 7) / 8);  // Each block does 8 outputs
        
        gemm_q4_0_fp32_m1_optimized<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            N, K
        );
    } else if (M <= 8) {
        // Small batch: shared memory for activation
        dim3 block(128);
        dim3 grid((N + 127) / 128, M);
        
        gemm_q4_0_fp32_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 64) {
        // Medium batch
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        
        gemm_q4_0_fp32_medium_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        
        gemm_q4_0_fp32_large_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM forward pass");
}
