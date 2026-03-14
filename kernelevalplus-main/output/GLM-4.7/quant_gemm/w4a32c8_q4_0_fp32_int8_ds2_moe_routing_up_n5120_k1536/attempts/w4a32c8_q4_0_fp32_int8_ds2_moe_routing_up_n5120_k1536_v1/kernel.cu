#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// BLOCK_Q4_0 format: 34 bytes per block of 32 values
// Layout: scale (fp16, 2 bytes) + qs (int8[32], 32 bytes)

// Helper function to unpack 4-bit values to int8
// Each int8 contains two 4-bit values (one in each nibble)
__device__ __inline__ int8_t get_q4_0_value(const int8_t* qs, int idx) {
    int byte_idx = idx / 2;
    int nibble = idx % 2;
    int8_t packed = qs[byte_idx];
    if (nibble == 0) {
        // Lower nibble (0-7)
        return (packed & 0x0F);
    } else {
        // Upper nibble (8-15)
        return (packed >> 4) & 0x0F;
    }
}

// Dot product of one BLOCK_Q4_0 weight block with one Q8_1 activation block
// Following llama.cpp vec_dot_q4_0_q8_1 pattern
__device__ __inline__ float vec_dot_q4_0_q8_1(
    const void* __restrict__ vw_packed,
    const float* __restrict__ act_block,
    int block_size
) {
    const int8_t* vw = (const int8_t*)vw_packed;
    const uint16_t* scale_ptr = (const uint16_t*)vw;  // First 2 bytes are scale (fp16)
    float scale_w = __half2float(*((const __half*)scale_ptr));
    const int8_t* qs = vw + 2;  // Quantized values start after scale

    // For activation, dynamically compute Q8_1-style quantization
    // Find max absolute value for scaling
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < block_size; i++) {
        max_abs = fmaxf(max_abs, fabsf(act_block[i]));
    }
    float scale_a = max_abs / 127.0f;

    // Compute dot product using int8 accumulation
    int32_t sumi = 0;
    #pragma unroll
    for (int i = 0; i < block_size; i++) {
        int8_t w_val = get_q4_0_value(qs, i);
        int8_t a_val = __float2int_rn(act_block[i] / fmaxf(scale_a, 1e-7f));
        // w_val in range [0, 15], need to shift to [-8, 7]
        w_val -= 8;
        sumi += w_val * a_val;
    }

    return scale_w * scale_a * (float)sumi;
}

// Optimized version using DP4A instruction (Compute Capability >= 6.1)
// Each DP4A computes dot product of 4 int8 pairs
__device__ __inline__ float vec_dot_q4_0_q8_1_dp4a(
    const void* __restrict__ vw_packed,
    const float* __restrict__ act_block,
    int block_size
) {
    const int8_t* vw = (const int8_t*)vw_packed;
    const uint16_t* scale_ptr = (const uint16_t*)vw;
    float scale_w = __half2float(*((const __half*)scale_ptr));
    const int8_t* qs = vw + 2;

    // Quantize activation to int8
    int8_t a_q[32];
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        max_abs = fmaxf(max_abs, fabsf(act_block[i]));
    }
    float scale_a = max_abs / 127.0f;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        a_q[i] = __float2int_rn(act_block[i] / fmaxf(scale_a, 1e-7f));
    }

    // Unpack Q4_0 weights and compute dot product using DP4A
    // Each int8 contains two Q4 values
    int32_t sumi = 0;

    // Process 8 bytes (16 Q4 values) at a time using 4 DP4A instructions
    // Each DP4A handles 4 int8 pairs = 8 Q4 values
    int32_t w_pairs[8];  // Unpacked weights

    // Manually unpack and compute using DP4A
    // Q4_0 format: values are in [0, 15], need to shift to [-8, 7]
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        int8_t packed = qs[i / 2];
        int8_t w0 = (packed & 0x0F) - 8;
        int8_t w1 = ((packed >> 4) & 0x0F) - 8;

        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;\n"
            : "+r"(sumi)
            : "r"(*(const int*)(&w0)), "r"(*(const int*)(&a_q[i])), "r"(sumi));
    }

    // Handle remaining values if block_size != 32
    for (int i = 16; i < 32; i++) {
        int8_t w_val = get_q4_0_value(qs, i) - 8;
        sumi += w_val * a_q[i];
    }

    return scale_w * scale_a * (float)sumi;
}

// Simpler version without DP4A for better compatibility
__device__ __inline__ float vec_dot_q4_0_fp32_simple(
    const void* __restrict__ vw_packed,
    const float* __restrict__ act_block,
    int block_size
) {
    const int8_t* vw = (const int8_t*)vw_packed;
    const uint16_t* scale_ptr = (const uint16_t*)vw;
    float scale_w = __half2float(*((const __half*)scale_ptr));
    const int8_t* qs = vw + 2;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < block_size; i++) {
        int8_t w_q4 = get_q4_0_value(qs, i);
        float w_val = (float)(w_q4 - 8) * scale_w;  // Q4_0 values are in [0, 15], center at 8
        sum += w_val * act_block[i];
    }

    return sum;
}

// Optimized kernel for compute-bound quantized GEMM
// Each thread block computes one row of output (M dimension)
// Each thread in block computes multiple output elements (N dimension)
__global__ void quant_gemm_q4_0_fp32_kernel(
    const void* __restrict__ weight,  // BLOCK_Q4_0 format: (N, K/32) blocks of 34 bytes
    const float* __restrict__ activation,  // (M, K)
    float* __restrict__ output,  // (M, N)
    int M, int N, int K
) {
    // blockDim.x = threads per block for N dimension
    // blockIdx.x = output row index (M dimension)
    // blockIdx.y = chunk of N dimension (for large N)

    const int m_idx = blockIdx.x;
    if (m_idx >= M) return;

    const int block_size = 32;  // Q4_0 block size
    const int n_chunks_per_block = blockDim.x;
    const int n_chunk = blockIdx.y * blockDim.x + threadIdx.x;
    const int n_idx = n_chunk * block_size;

    if (n_idx >= N) return;

    // Each thread computes one block of 32 output values
    const int n_end = min(n_idx + block_size, N);
    const int n_valid = n_end - n_idx;

    // Accumulators for this thread's portion of output
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }

    // Iterate over K dimension in blocks of 32
    const int k_blocks = K / block_size;

    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * block_size;

        // Load activation block for this row
        float act_block[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            act_block[i] = activation[m_idx * K + k_base + i];
        }

        // Process all N blocks for this K block
        // Each thread handles its assigned N block
        for (int nb = 0; nb < n_valid; nb++) {
            const int n_cur = n_idx + nb;

            // Find weight block for (n_cur, k_base)
            // Weight layout: (N, K/32) blocks
            const int weight_block_idx = n_cur * k_blocks + kb;
            const void* weight_block = (const char*)weight + weight_block_idx * 34;  // 34 bytes per Q4_0 block

            // Compute dot product
            float dot = vec_dot_q4_0_fp32_simple(weight_block, act_block, block_size);
            acc[nb] += dot;
        }
    }

    // Write results
    for (int nb = 0; nb < n_valid; nb++) {
        const int n_cur = n_idx + nb;
        output[m_idx * N + n_cur] = acc[nb];
    }
}

// Improved kernel with shared memory for activation tiling
__global__ void quant_gemm_q4_0_fp32_shared_kernel(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    if (m_idx >= M) return;

    const int block_size = 32;
    const int n_chunks_per_block = blockDim.x;
    const int n_chunk = blockIdx.y * blockDim.x + threadIdx.x;
    const int n_idx = n_chunk * block_size;

    if (n_idx >= N) return;

    const int n_end = min(n_idx + block_size, N);
    const int n_valid = n_end - n_idx;

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }

    const int k_blocks = K / block_size;

    // Shared memory for activation block (reused across K iterations)
    __shared__ float s_act[32];

    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * block_size;

        // Load activation into shared memory (coalesced access)
        if (threadIdx.x < 32) {
            s_act[threadIdx.x] = activation[m_idx * K + k_base + threadIdx.x];
        }
        __syncthreads();

        // Process weight blocks
        for (int nb = 0; nb < n_valid; nb++) {
            const int n_cur = n_idx + nb;
            const int weight_block_idx = n_cur * k_blocks + kb;
            const void* weight_block = (const char*)weight + weight_block_idx * 34;

            float dot = 0.0f;
            const int8_t* vw = (const int8_t*)weight_block;
            const uint16_t* scale_ptr = (const uint16_t*)vw;
            float scale_w = __half2float(*((const __half*)scale_ptr));
            const int8_t* qs = vw + 2;

            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int8_t w_q4 = get_q4_0_value(qs, i);
                float w_val = (float)(w_q4 - 8) * scale_w;
                dot += w_val * s_act[i];
            }

            acc[nb] += dot;
        }
        __syncthreads();
    }

    // Write results
    for (int nb = 0; nb < n_valid; nb++) {
        const int n_cur = n_idx + nb;
        output[m_idx * N + n_cur] = acc[nb];
    }
}

// Host wrapper function
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const int block_size = 32;
    const int n_blocks = (N + block_size - 1) / block_size;

    // Choose kernel based on size
    if (M == 1) {
        // For single row, use more threads per block
        const int threads_per_block = 256;
        const int n_chunks = (n_blocks + threads_per_block - 1) / threads_per_block;

        dim3 grid(M, n_chunks);
        dim3 block(threads_per_block);

        quant_gemm_q4_0_fp32_shared_kernel<<<grid, block>>>(
            weight.data_ptr(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // For multiple rows, balance occupancy
        const int threads_per_block = 128;
        const int n_chunks = (n_blocks + threads_per_block - 1) / threads_per_block;

        dim3 grid(M, n_chunks);
        dim3 block(threads_per_block);

        quant_gemm_q4_0_fp32_shared_kernel<<<grid, block>>>(
            weight.data_ptr(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation)");
}
