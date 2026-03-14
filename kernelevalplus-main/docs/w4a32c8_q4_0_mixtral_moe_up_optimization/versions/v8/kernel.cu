/**
 * W4A32C8: Q4_0 x FP32 GEMM - INT8 Tensor Core via DP4A (V8)
 *
 * Key Optimization: Use DP4A for INT8 dot products (330 TFLOPS peak)
 *
 * Q4_0 encoding: q ∈ [0, 15], decode: val = scale * (q - 8)
 * The (q-8) term gives INT8 values in [-8, 7] - perfect for DP4A!
 *
 * Formula: result = d4_0 * (d8_1 * sumi - 8 * s8_1)
 * Where:
 *   - d4_0: weight scale
 *   - d8_1: activation quantization scale (amax / 127)
 *   - sumi: INT8 dot product via DP4A
 *   - s8_1: sum of original activation values
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "");

__device__ __forceinline__ float half_to_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Pack 4 int8 values into a 32-bit integer for DP4A
__device__ __forceinline__ int pack_int8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (int)((uint8_t)a | ((uint8_t)b << 8) | ((uint8_t)c << 16) | ((uint8_t)d << 24));
}

// ============================================================================
// INT8 Tensor Core kernel using DP4A
// ============================================================================
__global__ void __launch_bounds__(128) gemm_dp4a_small_batch(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / 32;
    const int n_base = blockIdx.x * 128;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (n_base >= N || m >= M) return;
    
    // Shared memory for activation (16KB)
    __shared__ float act_shared[4096];
    // Shared memory for quantized activation INT8 values (4KB)
    __shared__ int8_t act_int8_shared[4096];
    // Shared memory for activation scales per block (512 bytes for 128 blocks)
    __shared__ float act_scale_shared[128];
    // Shared memory for activation sums per block (512 bytes)
    __shared__ float act_sum_shared[128];
    
    // Cooperative load of activation
    const float4* act_vec = reinterpret_cast<const float4*>(activation + m * K);
    float4* act_shared_vec = reinterpret_cast<float4*>(act_shared);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        if (idx < K / 4) act_shared_vec[idx] = act_vec[idx];
    }
    __syncthreads();
    
    // Compute quantization parameters per K-block
    // Each thread processes one K-block
    const int kb = tid;
    if (kb < num_blocks_k) {
        const float* act_ptr = act_shared + kb * 32;
        
        // Find max absolute value
        float amax = 0.0f;
        float sum_act = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float val = act_ptr[i];
            amax = fmaxf(amax, fabsf(val));
            sum_act += val;
        }
        
        // Q8_1 scale
        float d_act = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        act_scale_shared[kb] = d_act;
        act_sum_shared[kb] = sum_act;
        
        // Quantize activation to INT8
        int8_t* act_int8_ptr = act_int8_shared + kb * 32;
        float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 1.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float val = act_ptr[i];
            int q = (int)roundf(val * inv_scale);
            act_int8_ptr[i] = (int8_t)max(-128, min(127, q));
        }
    }
    __syncthreads();
    
    // Each thread computes 1 output
    const int n = n_base + tid;
    if (n >= N) return;
    
    const block_q4_0* w_row = (const block_q4_0*)weight_q + n * num_blocks_k;
    float sum = 0.0f;
    
    // Process all K blocks using DP4A
    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_0* wb = &w_row[kb];
        const float d_w = half_to_float(wb->d);
        const float d_a = act_scale_shared[kb];
        const float s_a = act_sum_shared[kb];
        const int8_t* act_int8 = act_int8_shared + kb * 32;
        
        // Unpack Q4_0 weights to INT8
        // qs[i] contains: low nibble = position i, high nibble = position i+16
        int sumi = 0;
        
        // Process 32 values using 8 DP4A operations
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Pack 4 weight INT8 values
            uint8_t p0 = wb->qs[i];
            uint8_t p1 = wb->qs[i + 8];
            int8_t w0 = (int8_t)((p0 & 0x0F) - 8);   // position i*4+0
            int8_t w1 = (int8_t)((p0 >> 4) - 8);     // position i*4+0+16
            int8_t w2 = (int8_t)((p1 & 0x0F) - 8);   // position i*4+2
            int8_t w3 = (int8_t)((p1 >> 4) - 8);     // position i*4+2+16
            
            int w_pack = pack_int8(w0, w1, w2, w3);
            int a_pack = pack_int8(act_int8[i*4], act_int8[i*4+16], 
                                   act_int8[i*4+2], act_int8[i*4+2+16]);
            
            sumi = __dp4a(w_pack, a_pack, sumi);
            
            // Also process the other 2 values per pair
            int8_t w4 = (int8_t)((wb->qs[i] & 0x0F) - 8);
            int8_t w5 = (int8_t)((wb->qs[i] >> 4) - 8);
            sumi += (int)w4 * (int)act_int8[i*4+1];
            sumi += (int)w5 * (int)act_int8[i*4+17];
        }
        
        // Actually, let me do a cleaner DP4A implementation
        // Reset and redo with proper packing
        sumi = 0;
        int8_t w_vals[32];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            w_vals[i] = (int8_t)((wb->qs[i] & 0x0F) - 8);
            w_vals[i + 16] = (int8_t)((wb->qs[i] >> 4) - 8);
        }
        
        // Use DP4A for 8 groups of 4
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_pack = *reinterpret_cast<int*>(w_vals + i * 4);
            int a_pack = *reinterpret_cast<const int*>(act_int8 + i * 4);
            sumi = __dp4a(w_pack, a_pack, sumi);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_pack = *reinterpret_cast<int*>(w_vals + 16 + i * 4);
            int a_pack = *reinterpret_cast<const int*>(act_int8 + 16 + i * 4);
            sumi = __dp4a(w_pack, a_pack, sumi);
        }
        
        // Apply formula: d_w * (d_a * sumi - 8 * s_a)
        sum += d_w * (d_a * (float)sumi - 8.0f * s_a);
    }
    
    output[m * N + n] = sum;
}

// ============================================================================
// Large batch kernel with DP4A
// ============================================================================
__global__ void __launch_bounds__(256) gemm_dp4a_large_batch(
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
        const block_q4_0* wb = &w_row[kb];
        const float d_w = half_to_float(wb->d);
        const float* act_ptr = act_row + kb * 32;
        
        // Compute activation quantization parameters
        float amax = 0.0f, s_a = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            amax = fmaxf(amax, fabsf(act_ptr[i]));
            s_a += act_ptr[i];
        }
        float d_a = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 1.0f;
        
        // Quantize activation to INT8
        int8_t a_vals[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int q = (int)roundf(act_ptr[i] * inv_scale);
            a_vals[i] = (int8_t)max(-128, min(127, q));
        }
        
        // Unpack weights to INT8
        int8_t w_vals[32];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            w_vals[i] = (int8_t)((wb->qs[i] & 0x0F) - 8);
            w_vals[i + 16] = (int8_t)((wb->qs[i] >> 4) - 8);
        }
        
        // DP4A dot product
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_pack = *reinterpret_cast<int*>(w_vals + i * 4);
            int a_pack = *reinterpret_cast<int*>(a_vals + i * 4);
            sumi = __dp4a(w_pack, a_pack, sumi);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w_pack = *reinterpret_cast<int*>(w_vals + 16 + i * 4);
            int a_pack = *reinterpret_cast<int*>(a_vals + 16 + i * 4);
            sumi = __dp4a(w_pack, a_pack, sumi);
        }
        
        sum += d_w * (d_a * (float)sumi - 8.0f * s_a);
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
    
    if (M <= 16) {
        dim3 block(128);
        dim3 grid((N + 127) / 128, M);
        gemm_dp4a_small_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    } else {
        dim3 block(256);
        dim3 grid((N + 255) / 256, M);
        gemm_dp4a_large_batch<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(), M, N, K);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM with DP4A");
}
