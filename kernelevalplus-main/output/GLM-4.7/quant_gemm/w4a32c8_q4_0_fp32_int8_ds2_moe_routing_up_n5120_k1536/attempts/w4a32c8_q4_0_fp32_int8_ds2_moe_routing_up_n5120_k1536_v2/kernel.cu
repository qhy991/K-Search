#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// BLOCK_Q4_0 format: 18 bytes per block
// - scale: FP16 (2 bytes)
// - qs: 32 packed 4-bit values in 16 bytes
//   Packing: byte[i] = q[i] | (q[i+16] << 4) for i in [0, 15]
//   Where q[i] in [0, 15] represents actual value in [-8, +7]

// Helper: Unpack Q4_0 packed value and convert to int8
// Returns value in [-8, +7] (Q4_0 encoding)
__device__ __inline__ int8_t unpack_q4_0(const uint8_t* packed, int idx) {
    if (idx < 16) {
        // Low nibbles: positions 0-15
        return (packed[idx] & 0x0F);
    } else {
        // High nibbles: positions 16-31
        return (packed[idx - 16] >> 4) & 0x0F;
    }
}

// Optimized dot product for one Q4_0 weight block with one activation block
// Following llama.cpp vec_dot_q4_0_q8_1 pattern adapted for FP32 activation
//
// Formula: d_w * (d_a * sumi - 8.0 * s_a)
// where:
//   d_w = weight scale (from Q4_0 block)
//   d_a = activation max / 127 (dynamic scale)
//   sumi = sum of (q_w * q_a) where q_w in [0,15], q_a in [-128,127]
//   s_a = sum of FP32 activation values (for offset compensation)
//
// The -8.0 * s_a term compensates for Q4_0's offset-8 encoding
__device__ __inline__ float vec_dot_q4_0_fp32_block(
    const uint8_t* __restrict__ weight_block,  // 18 bytes: scale (2) + packed qs (16)
    const float* __restrict__ act_block,        // 32 FP32 values
    int block_size
) {
    // Read weight scale (FP16)
    half d_w_half = *((const half*)weight_block);
    float d_w = __half2float(d_w_half);

    // Packed Q4_0 values start after scale
    const uint8_t* packed_qs = weight_block + 2;

    // Compute activation statistics for dynamic Q8_1 quantization
    float a_max = 0.0f;
    float a_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float a_val = act_block[i];
        float abs_val = fabsf(a_val);
        a_max = fmaxf(a_max, abs_val);
        a_sum += a_val;
    }
    float d_a = a_max / 127.0f;
    if (d_a < 1e-7f) d_a = 1e-7f;

    // Compute sumi using int8 accumulation
    // sumi = sum of (q_w * q_a) where q_w in [0,15], q_a in [-128,127]
    int32_t sumi = 0;

    // Unroll for better performance
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int8_t q_w = unpack_q4_0(packed_qs, i);  // [0, 15]
        int8_t q_a = __float2int_rn(act_block[i] / d_a);  // [-128, 127]
        sumi += q_w * q_a;
    }

    // Apply formula: d_w * (d_a * sumi - 8.0 * s_a)
    return d_w * (d_a * (float)sumi - 8.0f * a_sum);
}

// Main kernel: Q4_0 weight x FP32 activation GEMM
//
// Grid strategy:
// - blockIdx.x: M (batch/row dimension)
// - blockIdx.y: chunk of N (output features)
// - threadIdx.x: processes one block of 32 N values
//
// Each thread computes one block of 32 output elements
__global__ void quant_gemm_q4_0_fp32_kernel(
    const void* __restrict__ weight,     // Flat bytes: [N * K/32 * 18]
    const float* __restrict__ activation, // [M, K]
    float* __restrict__ output,          // [M, N]
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    if (m_idx >= M) return;

    const int block_size = 32;
    const int n_idx = (blockIdx.y * blockDim.x + threadIdx.x) * block_size;

    if (n_idx >= N) return;

    const int n_end = min(n_idx + block_size, N);
    const int k_blocks = K / block_size;

    // Accumulators for output values
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }

    // Process each K block
    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * block_size;

        // Load activation block for this row
        float act_block[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            act_block[i] = activation[m_idx * K + k_base + i];
        }

        // Process all N values for this K block
        for (int nb = 0; nb < 32 && (n_idx + nb) < N; nb++) {
            const int n_cur = n_idx + nb;

            // Calculate weight block offset
            // Weight layout: [N, k_blocks, 18]
            const int weight_block_idx = n_cur * k_blocks + kb;
            const uint8_t* weight_block = (const uint8_t*)weight + weight_block_idx * 18;

            // Compute dot product
            float dot = vec_dot_q4_0_fp32_block(weight_block, act_block, block_size);
            acc[nb] += dot;
        }
    }

    // Write results
    for (int nb = 0; nb < 32 && (n_idx + nb) < N; nb++) {
        output[m_idx * N + n_idx + nb] = acc[nb];
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

    // Optimize thread configuration based on M
    int threads_per_block;
    int min_grid_size;

    if (M == 1) {
        // Single batch: maximize threads per block
        threads_per_block = min(256, (int)(n_blocks * 32));
        threads_per_block = (threads_per_block + 31) / 32 * 32;  // Round to warp multiple
    } else {
        // Multiple batches: balance occupancy
        threads_per_block = 128;
    }

    const int n_chunks = (n_blocks + (threads_per_block / 32) - 1) / (threads_per_block / 32);

    dim3 grid(M, n_chunks);
    dim3 block(threads_per_block);

    quant_gemm_q4_0_fp32_kernel<<<grid, block>>>(
        weight.data_ptr(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation)");
}
