#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <limits>

// Constants for quantization
constexpr int BLOCK_Q4_0 = 32;
constexpr int BLOCK_Q8_1 = 32;

// DP4A helper - dot product of 4 int8 values
// Uses __dp4a instruction available on CC 6.1+
__device__ inline int dp4a(const int8_t* a, const int8_t* b) {
    int result;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(*reinterpret_cast<const int*>(a)), "r"(*reinterpret_cast<const int*>(b)), "r"(0));
    return result;
}

// Vectorized load of 32 int8 values
__device__ inline void load_q4_0_block(const uint8_t* __restrict__ weight_ptr, int8_t* __restrict__ qs, half* __restrict__ scale) {
    // Load scale (first 2 bytes as fp16)
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(weight_ptr);
    *scale = *reinterpret_cast<const half*>(&scale_bits);

    // Load 32 packed 4-bit values and unpack to 32 int8 values
    // Q4_0 stores 2 values per byte
    const uint8_t* packed = weight_ptr + 2;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t packed_val = packed[i];
        qs[2 * i] = (packed_val & 0x0F) - 8;  // Lower 4 bits, offset by -8
        qs[2 * i + 1] = (packed_val >> 4) - 8; // Upper 4 bits, offset by -8
    }
}

// Dynamic quantization of FP32 activation to Q8_1 style (scale only, no sum needed for dot product)
__device__ inline void quantize_activation_q8(const float* __restrict__ act_ptr, int8_t* __restrict__ qs, float* __restrict__ scale) {
    // Find max absolute value for scaling
    float max_val = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        max_val = fmaxf(max_val, fabsf(act_ptr[i]));
    }

    // Compute scale: scale = max_val / 127.0 (max int8)
    *scale = max_val / 127.0f;

    // Quantize to int8
    if (*scale > 1e-6f) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float rounded = roundf(act_ptr[i] / (*scale));
            qs[i] = static_cast<int8_t>(max(-128.0f, min(127.0f, rounded)));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            qs[i] = 0;
        }
    }
}

// Dot product between Q4_0 and Q8 blocks (32 values)
__device__ inline float dot_q4_0_q8(const int8_t* q4_vals, const int8_t* q8_vals, float scale_w, float scale_a) {
    // Use 8 dp4a instructions to compute dot product of 32 values
    // Each dp4a computes sum of 4 multiplications
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum += dp4a(q4_vals + 4 * i, q8_vals + 4 * i);
    }

    // Apply scales: result = scale_w * scale_a * sum
    return scale_w * scale_a * static_cast<float>(sum);
}

// Shared memory tile for weights and activations
template<int TILE_N, int TILE_K>
struct SharedTiles {
    half scale_w[TILE_N / 32][TILE_K / 32];  // One scale per block
    int8_t qs_w[TILE_N / 32][TILE_K];       // Unpacked Q4 weights (N/32 blocks, K values per block)
    int8_t qs_a[TILE_K / 32][TILE_K];       // Quantized activation (K/32 blocks, K values per block)
    float scale_a[TILE_K / 32];              // Per-block activation scales
};

// Main kernel for W4A32C8 Q4_0×Q8_1 GEMM
template<int TILE_N, int TILE_K>
__global__ void w4a32c8_q4_0_q8_1_gemm_kernel(
    const float* __restrict__ activation,   // [M, K]
    const uint8_t* __restrict__ weight,      // [N, K/32] packed Q4_0
    float* __restrict__ output,              // [M, N]
    const int M, const int N, const int K
) {
    // Grid-stride loop over rows
    const int row_m = blockIdx.x * TILE_N + threadIdx.x;

    if (row_m >= M) return;

    // Shared memory for tiles
    __shared__ SharedTiles<TILE_N, TILE_K> tiles;

    // Accumulators for output (N values per row)
    float acc[TILE_N];
    #pragma unroll
    for (int i = 0; i < TILE_N; ++i) {
        acc[i] = 0.0f;
    }

    // Number of K tiles
    const int num_k_tiles = K / TILE_K;

    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_K;

        // Each thread loads one activation block and quantizes it
        const int my_act_block = threadIdx.x / 8;  // Distribute among threads
        const int my_act_offset = k_base + my_act_block * 32;

        if (threadIdx.x < TILE_N) {
            // Quantize activation to Q8
            float scale_a_val;
            int8_t qs_a_val[32];
            const float* act_ptr = activation + row_m * K + my_act_offset;
            quantize_activation_q8(act_ptr, qs_a_val, &scale_a_val);

            // Store to shared memory
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                tiles.qs_a[my_act_block][i] = qs_a_val[i];
            }
            tiles.scale_a[my_act_block] = scale_a_val;
        }

        // Load weight tiles
        // Each thread loads one weight block
        const int weight_row = (blockIdx.y * TILE_N + threadIdx.x * 8) % N;
        const int weight_col_block = k_tile * (TILE_K / 32);

        __syncthreads();

        // Compute partial results
        #pragma unroll
        for (int i = 0; i < TILE_N; i += 8) {
            const int n_idx = (blockIdx.y * TILE_N + i + threadIdx.x * 8) % N;
            const int q4_block_idx = n_idx / 32;

            // Load Q4_0 weight block
            const uint8_t* weight_block_ptr = weight + n_idx * (K / 32) * 34 + (k_base / 32) * 34;
            half scale_w;
            int8_t qs_w[32];
            load_q4_0_block(weight_block_ptr, qs_w, &scale_w);

            // Dot product with each activation block
            float partial = 0.0f;
            #pragma unroll
            for (int b = 0; b < TILE_K / 32; ++b) {
                partial += dot_q4_0_q8(qs_w, &tiles.qs_a[b][0], __half2float(scale_w), tiles.scale_a[b]);
            }

            if (i + threadIdx.x * 8 < TILE_N) {
                acc[i + threadIdx.x * 8] += partial;
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < TILE_N; ++i) {
        const int n_idx = blockIdx.y * TILE_N + i + threadIdx.x * 8;
        if (n_idx < N && row_m < M) {
            output[row_m * N + n_idx] = acc[i];
        }
    }
}

// Simpler, more straightforward kernel using coalesced loads
__global__ void w4a32c8_q4_0_q8_1_gemm_kernel_simple(
    const float* __restrict__ activation,   // [M, K]
    const uint8_t* __restrict__ weight,      // [N, K/32] packed Q4_0
    float* __restrict__ output,              // [M, N]
    const int M, const int N, const int K
) {
    // Each thread computes one output element
    const int row_m = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_n = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_m >= M || col_n >= N) return;

    // Accumulator
    float acc = 0.0f;

    // Number of blocks in K dimension
    const int num_blocks = K / 32;

    // Each thread processes its own K blocks
    for (int b = 0; b < num_blocks; ++b) {
        // Load and quantize activation block (Q4_0 uses 32 values per block)
        int8_t qs_a[32];
        float scale_a;
        const float* act_ptr = activation + row_m * K + b * 32;
        quantize_activation_q8(act_ptr, qs_a, &scale_a);

        // Load Q4_0 weight block
        const uint8_t* weight_ptr = weight + col_n * (K / 32) * 34 + b * 34;
        half scale_w;
        int8_t qs_w[32];
        load_q4_0_block(weight_ptr, qs_w, &scale_w);

        // Dot product
        acc += dot_q4_0_q8(qs_w, qs_a, __half2float(scale_w), scale_a);
    }

    output[row_m * N + col_n] = acc;
}

// Host function to launch the kernel
torch::Tensor forward(
    torch::Tensor activation,
    torch::Tensor weight_q4_0,
    int N, int K
) {
    const int M = activation.size(0);

    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    // Choose kernel based on batch size
    if (M <= 8) {
        // For small batches, use simple coalesced kernel
        const int block_x = 64;
        const int block_y = 4;
        const dim3 grid((N + block_x - 1) / block_x, (M + block_y - 1) / block_y);
        const dim3 block(block_x, block_y);

        w4a32c8_q4_0_q8_1_gemm_kernel_simple<<<grid, block>>>(
            activation.data_ptr<float>(),
            weight_q4_0.data_ptr<uint8_t>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // For larger batches, use tiled kernel
        constexpr int TILE_N = 64;
        constexpr int TILE_K = 32;
        const dim3 grid((M + TILE_N - 1) / TILE_N, (N + TILE_N - 1) / TILE_N);
        const dim3 block(TILE_N);

        w4a32c8_q4_0_q8_1_gemm_kernel<TILE_N, TILE_K><<<grid, block>>>(
            activation.data_ptr<float>(),
            weight_q4_0.data_ptr<uint8_t>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0×Q8_1 GEMM");
}
