/**
 * W4A32C8: Q4_0 weight x FP32 activation GEMM kernel - Optimized V9
 *
 * DeepSeek-V2 MoE Routing Down projection: N=1536, K=5120
 *
 * Q4_0 format:
 *   - 18 bytes per block (2 bytes FP16 scale + 16 bytes packed 4-bit values)
 *   - Unpacking: llama.cpp style (all low nibbles 0-15, then high nibbles 16-31)
 *   - Dequantization: w = d_w × (q - 8), where q ∈ [0, 15]
 *
 * V9 Optimizations:
 *   - 32 threads per block (1 warp) for M=1 - maximum block count
 *   - Each block processes exactly one N value
 *   - No warp reduction needed (single warp)
 *   - Loop unrolling
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define QK 32
#define BLOCK_Q4_0_SIZE 18
#define WARP_SIZE 32

// ============================================================================
// Helper: Decode FP16 from bytes
// ============================================================================
__device__ __forceinline__ float half_to_float_fast(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// ============================================================================
// Kernel: M=1 - 32 threads (1 warp) per block
// Each block handles one N value, no warp reduction needed
// ============================================================================
__global__ void __launch_bounds__(32) gemm_q4_0_fp32_m1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int N, const int K
) {
    const int num_blocks_k = K / QK;
    const int lane_id = threadIdx.x;
    const int n = blockIdx.x;

    if (n >= N) return;

    const float* act_row = activation;
    const uint8_t* w_row = weight + (long long)n * num_blocks_k * BLOCK_Q4_0_SIZE;

    float sum = 0.0f;

    // Process all K blocks with unrolling
    int kb = 0;
    for (; kb + 7 < num_blocks_k; kb += 8) {
        const uint8_t* w_block0 = w_row + (kb + 0) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block1 = w_row + (kb + 1) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block2 = w_row + (kb + 2) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block3 = w_row + (kb + 3) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block4 = w_row + (kb + 4) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block5 = w_row + (kb + 5) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block6 = w_row + (kb + 6) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block7 = w_row + (kb + 7) * BLOCK_Q4_0_SIZE;

        const float* act_block0 = act_row + (kb + 0) * QK;
        const float* act_block1 = act_row + (kb + 1) * QK;
        const float* act_block2 = act_row + (kb + 2) * QK;
        const float* act_block3 = act_row + (kb + 3) * QK;
        const float* act_block4 = act_row + (kb + 4) * QK;
        const float* act_block5 = act_row + (kb + 5) * QK;
        const float* act_block6 = act_row + (kb + 6) * QK;
        const float* act_block7 = act_row + (kb + 7) * QK;

        float d_w0 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block0));
        float d_w1 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block1));
        float d_w2 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block2));
        float d_w3 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block3));
        float d_w4 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block4));
        float d_w5 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block5));
        float d_w6 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block6));
        float d_w7 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block7));

        const uint8_t* packed0 = w_block0 + 2;
        const uint8_t* packed1 = w_block1 + 2;
        const uint8_t* packed2 = w_block2 + 2;
        const uint8_t* packed3 = w_block3 + 2;
        const uint8_t* packed4 = w_block4 + 2;
        const uint8_t* packed5 = w_block5 + 2;
        const uint8_t* packed6 = w_block6 + 2;
        const uint8_t* packed7 = w_block7 + 2;

        int q0, q1, q2, q3, q4, q5, q6, q7;
        float a0, a1, a2, a3, a4, a5, a6, a7;

        if (lane_id < 16) {
            q0 = (packed0[lane_id] & 0x0F);
            q1 = (packed1[lane_id] & 0x0F);
            q2 = (packed2[lane_id] & 0x0F);
            q3 = (packed3[lane_id] & 0x0F);
            q4 = (packed4[lane_id] & 0x0F);
            q5 = (packed5[lane_id] & 0x0F);
            q6 = (packed6[lane_id] & 0x0F);
            q7 = (packed7[lane_id] & 0x0F);
            a0 = act_block0[lane_id];
            a1 = act_block1[lane_id];
            a2 = act_block2[lane_id];
            a3 = act_block3[lane_id];
            a4 = act_block4[lane_id];
            a5 = act_block5[lane_id];
            a6 = act_block6[lane_id];
            a7 = act_block7[lane_id];
        } else {
            q0 = ((packed0[lane_id - 16] >> 4) & 0x0F);
            q1 = ((packed1[lane_id - 16] >> 4) & 0x0F);
            q2 = ((packed2[lane_id - 16] >> 4) & 0x0F);
            q3 = ((packed3[lane_id - 16] >> 4) & 0x0F);
            q4 = ((packed4[lane_id - 16] >> 4) & 0x0F);
            q5 = ((packed5[lane_id - 16] >> 4) & 0x0F);
            q6 = ((packed6[lane_id - 16] >> 4) & 0x0F);
            q7 = ((packed7[lane_id - 16] >> 4) & 0x0F);
            a0 = act_block0[lane_id];
            a1 = act_block1[lane_id];
            a2 = act_block2[lane_id];
            a3 = act_block3[lane_id];
            a4 = act_block4[lane_id];
            a5 = act_block5[lane_id];
            a6 = act_block6[lane_id];
            a7 = act_block7[lane_id];
        }

        sum += a0 * d_w0 * static_cast<float>(q0 - 8);
        sum += a1 * d_w1 * static_cast<float>(q1 - 8);
        sum += a2 * d_w2 * static_cast<float>(q2 - 8);
        sum += a3 * d_w3 * static_cast<float>(q3 - 8);
        sum += a4 * d_w4 * static_cast<float>(q4 - 8);
        sum += a5 * d_w5 * static_cast<float>(q5 - 8);
        sum += a6 * d_w6 * static_cast<float>(q6 - 8);
        sum += a7 * d_w7 * static_cast<float>(q7 - 8);
    }

    // Process remaining blocks
    for (; kb < num_blocks_k; kb++) {
        const uint8_t* w_block = w_row + kb * BLOCK_Q4_0_SIZE;
        const float* act_block = act_row + kb * QK;

        float d_w = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* packed = w_block + 2;

        int q;
        float a;
        if (lane_id < 16) {
            q = (packed[lane_id] & 0x0F);
            a = act_block[lane_id];
        } else {
            q = ((packed[lane_id - 16] >> 4) & 0x0F);
            a = act_block[lane_id];
        }

        sum += a * d_w * static_cast<float>(q - 8);
    }

    // Warp reduction (needed since we compute partial sums)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
        output[n] = sum;
    }
}

// ============================================================================
// Kernel: M>1 - Optimized with 128 threads
// ============================================================================
__global__ void __launch_bounds__(128) gemm_q4_0_fp32_mlarge_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    const int num_blocks_k = K / QK;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    const int n = blockIdx.x * 4 + warp_id;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    const float* act_row = activation + (long long)m * K;
    const uint8_t* w_row = weight + (long long)n * num_blocks_k * BLOCK_Q4_0_SIZE;

    float sum = 0.0f;

    // Process blocks with unrolling
    int kb = 0;
    for (; kb + 3 < num_blocks_k; kb += 4) {
        const uint8_t* w_block0 = w_row + (kb + 0) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block1 = w_row + (kb + 1) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block2 = w_row + (kb + 2) * BLOCK_Q4_0_SIZE;
        const uint8_t* w_block3 = w_row + (kb + 3) * BLOCK_Q4_0_SIZE;

        const float* act_block0 = act_row + (kb + 0) * QK;
        const float* act_block1 = act_row + (kb + 1) * QK;
        const float* act_block2 = act_row + (kb + 2) * QK;
        const float* act_block3 = act_row + (kb + 3) * QK;

        float d_w0 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block0));
        float d_w1 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block1));
        float d_w2 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block2));
        float d_w3 = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block3));

        const uint8_t* packed0 = w_block0 + 2;
        const uint8_t* packed1 = w_block1 + 2;
        const uint8_t* packed2 = w_block2 + 2;
        const uint8_t* packed3 = w_block3 + 2;

        int q0, q1, q2, q3;
        float a0, a1, a2, a3;

        if (lane_id < 16) {
            q0 = (packed0[lane_id] & 0x0F);
            q1 = (packed1[lane_id] & 0x0F);
            q2 = (packed2[lane_id] & 0x0F);
            q3 = (packed3[lane_id] & 0x0F);
            a0 = act_block0[lane_id];
            a1 = act_block1[lane_id];
            a2 = act_block2[lane_id];
            a3 = act_block3[lane_id];
        } else {
            q0 = ((packed0[lane_id - 16] >> 4) & 0x0F);
            q1 = ((packed1[lane_id - 16] >> 4) & 0x0F);
            q2 = ((packed2[lane_id - 16] >> 4) & 0x0F);
            q3 = ((packed3[lane_id - 16] >> 4) & 0x0F);
            a0 = act_block0[lane_id];
            a1 = act_block1[lane_id];
            a2 = act_block2[lane_id];
            a3 = act_block3[lane_id];
        }

        sum += a0 * d_w0 * static_cast<float>(q0 - 8);
        sum += a1 * d_w1 * static_cast<float>(q1 - 8);
        sum += a2 * d_w2 * static_cast<float>(q2 - 8);
        sum += a3 * d_w3 * static_cast<float>(q3 - 8);
    }

    // Process remaining blocks
    for (; kb < num_blocks_k; kb++) {
        const uint8_t* w_block = w_row + kb * BLOCK_Q4_0_SIZE;
        const float* act_block = act_row + kb * QK;

        float d_w = half_to_float_fast(*reinterpret_cast<const uint16_t*>(w_block));
        const uint8_t* packed = w_block + 2;

        int q;
        float a;
        if (lane_id < 16) {
            q = (packed[lane_id] & 0x0F);
            a = act_block[lane_id];
        } else {
            q = ((packed[lane_id - 16] >> 4) & 0x0F);
            a = act_block[lane_id];
        }

        sum += a * d_w * static_cast<float>(q - 8);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
        output[(long long)m * N + n] = sum;
    }
}

// ============================================================================
// Host wrapper
// ============================================================================
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if (M == 1) {
        const int threads = 32;
        const int grid = N;  // One block per N value

        gemm_q4_0_fp32_m1_kernel<<<grid, threads>>>(
            weight_ptr, act_ptr, output_ptr, N, K);
    } else {
        const int threads = 128;
        const int grid_n = (N + 3) / 4;

        dim3 grid(grid_n, M);
        dim3 block(threads);

        gemm_q4_0_fp32_mlarge_kernel<<<grid, block>>>(
            weight_ptr, act_ptr, output_ptr, M, N, K);
    }

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
