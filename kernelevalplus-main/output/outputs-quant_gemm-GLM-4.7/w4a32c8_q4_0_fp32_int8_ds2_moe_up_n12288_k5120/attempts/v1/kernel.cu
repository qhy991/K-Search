#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// BLOCK_Q4_0 format: 18 bytes per 32 values
// Bytes 0-1: FP16 scale (d)
// Bytes 2-17: 16 bytes, each containing 2 packed 4-bit values
// Packing: byte[i] = qs[i] (low nibble) | qs[i+16] (high nibble) << 4
// Decoding: val = d * (qs - 8), where qs is unpacked to [0, 15]

// Q8_1-style dynamic activation quantization
// Each block of 32 activation values is quantized on-the-fly
// Scale d_a = max(abs(values)) / 127.0
// Sum s_a = sum of original FP32 values
// Quantized: a_qs = round(a / d_a), clamped to [-128, 127]

// Computation formula per block:
// sumi = dot(w_qs[i], a_qs[i]) for i in [0, 31]
// output += d_w * (d_a * sumi - 8 * s_a)

// Convert FP16 to FP32
__device__ __forceinline__ float fp16_to_fp32(unsigned short fp16) {
    unsigned int fp32 = ((fp16 & 0x8000) << 16) | (((fp16 & 0x7c00) + 0x1C000) << 13) | ((fp16 & 0x03FF) << 13);
    return __int_as_float(fp32);
}

// Unpack 4 Q4_0 values from 2 bytes
// Each byte contains 2 4-bit values: low nibble = qs[i], high nibble = qs[i+16]
__device__ __forceinline__ int4 unpack_q4_0_4values(const uint8_t* packed) {
    int4 result;
    // Low nibbles: positions 0-3
    result.x = packed[0] & 0x0F;
    result.y = packed[1] & 0x0F;
    result.z = packed[2] & 0x0F;
    result.w = packed[3] & 0x0F;
    return result;
}

// Kernel for small M (1-8): Memory-bound, vectorized loads
// One warp per N output element, process all K blocks
__global__ void w4a32c8_q4_0_kernel_small_batch(
    const uint8_t* __restrict__ weight,      // Q4_0 weights: [N, K/32] blocks of 18 bytes
    const float* __restrict__ activation,    // FP32 activation: [M, K]
    float* __restrict__ output,              // Output: [M, N]
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;

    // One warp per N output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int n = warp_id;

    if (n >= N) return;

    // Each warp processes all M rows
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        const float* act_row = activation + m * K;

        // Process all blocks
        for (int kb = 0; kb < K_BLOCKS; kb++) {
            // Load activation block (32 values) and compute d_a, s_a
            float act_block[32];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                act_block[i] = act_row[kb * 32 + i];
            }

            // Compute d_a (scale) and s_a (sum)
            float d_a = 0.0f;
            float s_a = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float abs_val = fabsf(act_block[i]);
                if (abs_val > d_a) d_a = abs_val;
                s_a += act_block[i];
            }
            d_a = fmaxf(d_a / 127.0f, 1e-6f);

            // Quantize activation to INT8
            int8_t a_qs[32];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int q = (int)roundf(act_block[i] / d_a);
                a_qs[i] = (int8_t)min(max(q, -128), 127);
            }

            // Load weight block and compute sumi
            const uint8_t* w_block = weight + (n * K_BLOCKS + kb) * 18;
            float d_w = fp16_to_fp32(*(unsigned short*)w_block);

            // Unpack Q4_0 weights and compute dot product
            const uint8_t* w_packed = w_block + 2;
            int sumi = 0;

            // Process all 32 values: 8 iterations of 4 values
            for (int i = 0; i < 8; i++) {
                // Unpack 4 values from 2 bytes (low nibbles)
                int4 w_vals_low;
                w_vals_low.x = w_packed[i] & 0x0F;
                w_vals_low.y = w_packed[i + 8] & 0x0F;
                w_vals_low.z = w_packed[i] & 0x0F;
                w_vals_low.w = w_packed[i + 8] & 0x0F;

                // Actually need proper unpacking
                // Byte layout: byte[i] = qs[i] | (qs[i+16] << 4)
                int8_t w0 = w_packed[i] & 0x0F;
                int8_t w1 = w_packed[i + 8] & 0x0F;
                int8_t w2 = (w_packed[i] >> 4) & 0x0F;
                int8_t w3 = (w_packed[i + 8] >> 4) & 0x0F;

                sumi += (int)a_qs[i * 4 + 0] * (w0 - 8);
                sumi += (int)a_qs[i * 4 + 1] * (w1 - 8);
                sumi += (int)a_qs[i * 4 + 2] * (w2 - 8);
                sumi += (int)a_qs[i * 4 + 3] * (w3 - 8);
            }

            // Apply formula: d_w * (d_a * sumi - 8 * s_a)
            acc += d_w * (d_a * (float)sumi - 8.0f * s_a);
        }

        output[m * N + n] = acc;
    }
}

// Kernel for larger M: Compute-bound, shared memory tiling
__global__ void w4a32c8_q4_0_kernel_large_batch(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCKS = K / 32;

    // Grid-stride loop over N
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;

    int tx = threadIdx.x;
    const int TILE_K = 4;  // Process 4 blocks per iteration (128 values)

    // Shared memory for weight blocks
    __shared__ float s_d_w[TILE_K];  // Weight scales
    __shared__ int8_t s_w_qs[TILE_K][32];  // Weight quantized values

    // Each thread processes one M row
    for (int m = 0; m < M; m++) {
        const float* act_row = activation + m * K;
        float acc = 0.0f;

        // Process blocks in tiles
        for (int kb_tile = 0; kb_tile < K_BLOCKS; kb_tile += TILE_K) {
            // Load weight blocks into shared memory
            #pragma unroll
            for (int i = 0; i < TILE_K; i++) {
                int kb = kb_tile + i;
                if (kb < K_BLOCKS) {
                    const uint8_t* w_block = weight + (n * K_BLOCKS + kb) * 18;
                    s_d_w[i] = fp16_to_fp32(*(unsigned short*)w_block);

                    // Unpack Q4_0 values
                    const uint8_t* w_packed = w_block + 2;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        int8_t w0 = w_packed[j] & 0x0F;
                        int8_t w1 = w_packed[j + 8] & 0x0F;
                        int8_t w2 = (w_packed[j] >> 4) & 0x0F;
                        int8_t w3 = (w_packed[j + 8] >> 4) & 0x0F;
                        s_w_qs[i][j * 4 + 0] = w0 - 8;
                        s_w_qs[i][j * 4 + 1] = w1 - 8;
                        s_w_qs[i][j * 4 + 2] = w2 - 8;
                        s_w_qs[i][j * 4 + 3] = w3 - 8;
                    }
                }
            }

            __syncthreads();

            // Compute dot products
            #pragma unroll
            for (int i = 0; i < TILE_K; i++) {
                int kb = kb_tile + i;
                if (kb < K_BLOCKS) {
                    // Load activation block and compute d_a, s_a
                    float act_block[32];
                    float d_a = 0.0f;
                    float s_a = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        act_block[j] = act_row[kb * 32 + j];
                        float abs_val = fabsf(act_block[j]);
                        if (abs_val > d_a) d_a = abs_val;
                        s_a += act_block[j];
                    }
                    d_a = fmaxf(d_a / 127.0f, 1e-6f);

                    // Compute sumi
                    int sumi = 0;
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        int8_t a_q = (int8_t)min(max((int)roundf(act_block[j] / d_a), -128), 127);
                        sumi += (int)a_q * (int)s_w_qs[i][j];
                    }

                    acc += s_d_w[i] * (d_a * (float)sumi - 8.0f * s_a);
                }
            }

            __syncthreads();
        }

        output[m * N + n] = acc;
    }
}

// Host function to dispatch kernels
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* activation_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if (M <= 8) {
        // Small batch: use memory-optimized kernel
        int warps_per_block = 8;  // 256 threads
        int threads_per_block = warps_per_block * 32;
        int num_blocks = (N + warps_per_block - 1) / warps_per_block;

        w4a32c8_q4_0_kernel_small_batch<<<num_blocks, threads_per_block>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    } else {
        // Large batch: use compute-optimized kernel
        dim3 threads(32, 4);  // 128 threads per block
        int num_blocks = (N + 3) / 4;

        w4a32c8_q4_0_kernel_large_batch<<<num_blocks, threads>>>(
            weight_ptr, activation_ptr, output_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
