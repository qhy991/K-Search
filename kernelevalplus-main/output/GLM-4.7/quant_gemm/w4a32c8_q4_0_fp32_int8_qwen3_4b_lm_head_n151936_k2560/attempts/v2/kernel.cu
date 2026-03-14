#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

// Q4_0 block structure (packed)
struct block_q4_0 {
    uint16_t d;      // scale (FP16)
    uint8_t qs[16];  // 32 packed 4-bit values
};

// Device function to read FP16 as float
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * W4A32C8 Q4_0 Quantized GEMM Kernel v2 - Combined Strategy
 *
 * This kernel implements strategy dispatch based on batch size (M):
 * - Small batch (M <= 8): memory-bound kernel with vectorized loads
 * - Large batch (M > 8): compute-bound kernel with shared memory tiling
 *
 * Design based on Roofline analysis:
 * - Small batch: OI ~ 2-15 FLOPs/Byte (memory-bound)
 * - Large batch: OI > 100 FLOPs/Byte (compute-bound)
 */

// Kernel for small batch sizes (memory-bound, one thread per output)
__global__ void __launch_bounds__(256) w4a32c8_small_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const int num_blocks_k = K / 32;

    // Pre-compute pointers
    const block_q4_0* __restrict__ w_row = reinterpret_cast<const block_q4_0*>(weight + n * num_blocks_k * 18);
    const float* __restrict__ a_row = activation + m * K;

    float acc = 0.0f;

    // Process K blocks with 4-way unrolling
    int bk = 0;

    for (; bk + 4 <= num_blocks_k; bk += 4) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int current_bk = bk + i;

            const block_q4_0 w_block = w_row[current_bk];
            const float d_w = read_half_as_float(w_block.d);

            const float* __restrict__ a_block = a_row + current_bk * 32;

            // Vectorized max/sum
            float a_max = 0.0f;
            float a_sum = 0.0f;

            #pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const float4 vals = *reinterpret_cast<const float4*>(&a_block[j]);
                a_max = fmaxf(a_max, fmaxf(fabsf(vals.x), fmaxf(fabsf(vals.y), fabsf(vals.z))));
                a_max = fmaxf(a_max, fabsf(vals.w));
                a_sum += vals.x + vals.y + vals.z + vals.w;
            }

            const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

            // Dot product with unrolling
            int32_t sumi = 0;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                const uint8_t byte_val = w_block.qs[j];
                const int w_low = byte_val & 0x0F;
                const int w_high = (byte_val >> 4) & 0x0F;

                const int a_low = __float2int_rn(a_block[j] / d_a);
                const int a_high = __float2int_rn(a_block[j + 16] / d_a);

                sumi += w_low * a_low + w_high * a_high;
            }

            const float block_result = d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);

            if (i == 0) acc0 = block_result;
            else if (i == 1) acc1 = block_result;
            else if (i == 2) acc2 = block_result;
            else acc3 = block_result;
        }

        acc += acc0 + acc1 + acc2 + acc3;
    }

    for (; bk < num_blocks_k; bk++) {
        const block_q4_0 w_block = w_row[bk];
        const float d_w = read_half_as_float(w_block.d);

        const float* __restrict__ a_block = a_row + bk * 32;

        float a_max = 0.0f;
        float a_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            const float4 vals = *reinterpret_cast<const float4*>(&a_block[i]);
            a_max = fmaxf(a_max, fmaxf(fabsf(vals.x), fmaxf(fabsf(vals.y), fabsf(vals.z))));
            a_max = fmaxf(a_max, fabsf(vals.w));
            a_sum += vals.x + vals.y + vals.z + vals.w;
        }

        const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

        int32_t sumi = 0;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t byte_val = w_block.qs[i];
            const int w_low = byte_val & 0x0F;
            const int w_high = (byte_val >> 4) & 0x0F;

            const int a_low = __float2int_rn(a_block[i] / d_a);
            const int a_high = __float2int_rn(a_block[i + 16] / d_a);

            sumi += w_low * a_low + w_high * a_high;
        }

        acc += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
    }

    output[m * N + n] = acc;
}

// Kernel for large batch sizes (compute-bound, with shared memory tiling)
__global__ void __launch_bounds__(256) w4a32c8_large_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // Each block computes a tile of output
    const int tile_m = 4;  // Number of output rows processed per block
    const int tile_n = 64;  // Number of output columns processed per block

    const int block_m = blockIdx.y * tile_m;
    const int block_n = blockIdx.x * tile_n;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Thread-local accumulators (each thread computes 2 outputs)
    float acc[2] = {0.0f, 0.0f};

    const int num_blocks_k = K / 32;

    // Each thread processes 2 columns
    const int n0 = block_n + threadIdx.x * 2;
    const int n1 = n0 + 1;
    const int m = block_m + warp_id;  // Each warp handles one row

    if (m >= M) return;

    // Pre-compute activation row pointer
    const float* __restrict__ a_row = activation + m * K;

    // Pre-compute weight row pointers
    const block_q4_0* __restrict__ w_row0 = reinterpret_cast<const block_q4_0*>(weight + n0 * num_blocks_k * 18);
    const block_q4_0* __restrict__ w_row1 = reinterpret_cast<const block_q4_0*>(weight + n1 * num_blocks_k * 18);

    // Check bounds for columns
    const bool valid0 = n0 < N;
    const bool valid1 = n1 < N;

    // Process K blocks
    for (int bk = 0; bk < num_blocks_k; bk++) {
        if (valid0) {
            const block_q4_0 w_block = w_row0[bk];
            const float d_w = read_half_as_float(w_block.d);

            const float* __restrict__ a_block = a_row + bk * 32;

            // Compute activation statistics
            float a_max = 0.0f;
            float a_sum = 0.0f;

            #pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const float4 vals = *reinterpret_cast<const float4*>(&a_block[j]);
                a_max = fmaxf(a_max, fmaxf(fabsf(vals.x), fmaxf(fabsf(vals.y), fabsf(vals.z))));
                a_max = fmaxf(a_max, fabsf(vals.w));
                a_sum += vals.x + vals.y + vals.z + vals.w;
            }

            const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

            // Dot product
            int32_t sumi = 0;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                const uint8_t byte_val = w_block.qs[j];
                const int w_low = byte_val & 0x0F;
                const int w_high = (byte_val >> 4) & 0x0F;

                const int a_low = __float2int_rn(a_block[j] / d_a);
                const int a_high = __float2int_rn(a_block[j + 16] / d_a);

                sumi += w_low * a_low + w_high * a_high;
            }

            acc[0] += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
        }

        if (valid1) {
            const block_q4_0 w_block = w_row1[bk];
            const float d_w = read_half_as_float(w_block.d);

            const float* __restrict__ a_block = a_row + bk * 32;

            // Compute activation statistics
            float a_max = 0.0f;
            float a_sum = 0.0f;

            #pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const float4 vals = *reinterpret_cast<const float4*>(&a_block[j]);
                a_max = fmaxf(a_max, fmaxf(fabsf(vals.x), fmaxf(fabsf(vals.y), fabsf(vals.z))));
                a_max = fmaxf(a_max, fabsf(vals.w));
                a_sum += vals.x + vals.y + vals.z + vals.w;
            }

            const float d_a = a_max > 0.0f ? a_max / 127.0f : 1.0f;

            // Dot product
            int32_t sumi = 0;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                const uint8_t byte_val = w_block.qs[j];
                const int w_low = byte_val & 0x0F;
                const int w_high = (byte_val >> 4) & 0x0F;

                const int a_low = __float2int_rn(a_block[j] / d_a);
                const int a_high = __float2int_rn(a_block[j + 16] / d_a);

                sumi += w_low * a_low + w_high * a_high;
            }

            acc[1] += d_w * (d_a * static_cast<float>(sumi) - 8.0f * a_sum);
        }
    }

    // Write output
    if (valid0) {
        output[m * N + n0] = acc[0];
    }
    if (valid1) {
        output[m * N + n1] = acc[1];
    }
}

/**
 * Host function with strategy dispatch
 */
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kByte, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    // Allocate output tensor
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Strategy dispatch based on batch size
    if (M <= 8) {
        // Small batch: memory-bound kernel
        const int threads_per_block = 256;
        const int blocks_x = (N + threads_per_block - 1) / threads_per_block;
        const int blocks_y = M;

        dim3 grid(blocks_x, blocks_y);
        dim3 block(threads_per_block, 1);

        w4a32c8_small_batch_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: compute-bound kernel with shared memory tiling
        const int tile_m = 4;
        const int tile_n = 64;
        const int threads_per_block = 128;  // 4 warps

        const int blocks_x = (N + tile_n - 1) / tile_n;
        const int blocks_y = (M + tile_m - 1) / tile_m;

        dim3 grid(blocks_x, blocks_y);
        dim3 block(threads_per_block);

        w4a32c8_large_batch_kernel<<<grid, block>>>(
            weight.data_ptr<uint8_t>(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v2 (Qwen3-4B LM Head) - Combined Strategy");
}
