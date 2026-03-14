#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// BLOCK_Q4_0 format: 18 bytes per block
// scale (uint16_t/fp16) + 16 bytes packed quanta (32 x 4-bit values)
typedef struct {
    uint16_t d;      // scale (fp16)
    uint8_t qs[16];  // packed quanta (32 x 4-bit values)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size must be 18 bytes");

// Helper to read fp16 scale as float32
__device__ inline float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Helper: unpack 4-bit values from packed bytes
// Each byte contains 2 x 4-bit values (lower nibble first)
__device__ inline int8_t unpack_q4_0(const uint8_t* qs, int idx) {
    int byte_idx = idx / 2;
    int nibble = idx % 2;
    uint8_t packed = qs[byte_idx];
    uint8_t q4 = (nibble == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    // Q4_0 uses offset -8 for unsigned storage
    return static_cast<int8_t>(static_cast<int32_t>(q4) - 8);
}

// Strategy 1: Simple bandwidth-optimized kernel for small batches
// Each thread computes one output element
__global__ void gemm_q4_0_fp32_simple(
    const uint8_t* __restrict__ weight,  // Packed block_q4_0 array
    const float* __restrict__ activation,  // [M, K]
    float* __restrict__ output,            // [M, N]
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;

    // Output position
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sum = 0.0f;

    // Iterate over K dimension in blocks of 32
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Load activation block (32 values) and quantize to Q8_1 style
        float a_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = activation[m * K + kb * 32 + i];
            a_max = fmaxf(a_max, fabsf(a_val));
        }

        float a_scale = a_max / 127.0f;
        if (a_scale < 1e-7f) a_scale = 1e-7f;

        // Load weight block
        const block_q4_0* w_block = &w_blocks[n * K_BLOCKS + kb];
        float w_scale = read_half_as_float(w_block->d);

        // Dot product of quantized values
        int32_t dot_i = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float a_val = activation[m * K + kb * 32 + i];
            int8_t a_q = static_cast<int8_t>(roundf(__float2int_rn(a_val / a_scale)));
            int8_t w_q = unpack_q4_0(w_block->qs, i);
            dot_i += static_cast<int32_t>(a_q) * static_cast<int32_t>(w_q);
        }

        sum += a_scale * w_scale * static_cast<float>(dot_i);
    }

    output[m * N + n] = sum;
}

// Strategy 2: Shared memory tiled kernel for large batches
__global__ void gemm_q4_0_fp32_tiled(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_BLOCK = 32;
    const int K_BLOCKS = K / K_BLOCK;

    // Tile dimensions
    const int TILE_M = 4;
    const int TILE_N = 32;

    // Shared memory for tiles
    __shared__ float s_activation[TILE_M * K_BLOCK];
    __shared__ float s_weight_scale[TILE_N];
    __shared__ int8_t s_weight_q[TILE_N * K_BLOCK];

    int tile_m = blockIdx.y * TILE_M;
    int tile_n = blockIdx.x * TILE_N;

    int lane_m = threadIdx.y;
    int lane_n = threadIdx.x;

    int m = tile_m + lane_m;
    int n = tile_n + lane_n;

    if (m >= M || n >= N) {
        if (m < M && n < N) output[m * N + n] = 0.0f;
        return;
    }

    const block_q4_0* w_blocks = reinterpret_cast<const block_q4_0*>(weight);

    float sum = 0.0f;

    // Iterate over K blocks
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        // Load activation tile cooperatively
        if (lane_n < K_BLOCK && (tile_m + lane_m) < M) {
            int a_idx = (tile_m + lane_m) * K + kb * K_BLOCK + lane_n;
            s_activation[lane_m * K_BLOCK + lane_n] = activation[a_idx];
        }

        // Load weight scales
        if (tile_n + lane_n < N) {
            const block_q4_0* w_block = &w_blocks[(tile_n + lane_n) * K_BLOCKS + kb];
            s_weight_scale[lane_n] = read_half_as_float(w_block->d);
        }

        // Load weight quanta
        if (tile_n + lane_n < N) {
            const block_q4_0* w_block = &w_blocks[(tile_n + lane_n) * K_BLOCKS + kb];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                // Unpack 2 values per byte
                s_weight_q[lane_n * K_BLOCK + i * 2] = unpack_q4_0(w_block->qs, i * 2);
                s_weight_q[lane_n * K_BLOCK + i * 2 + 1] = unpack_q4_0(w_block->qs, i * 2 + 1);
            }
        }

        __syncthreads();

        // Compute dot product
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < K_BLOCK; ++i) {
            a_max = fmaxf(a_max, fabsf(s_activation[lane_m * K_BLOCK + i]));
        }
        float a_scale = a_max / 127.0f;
        if (a_scale < 1e-7f) a_scale = 1e-7f;

        int32_t dot_i = 0;
        #pragma unroll
        for (int i = 0; i < K_BLOCK; ++i) {
            float a_val = s_activation[lane_m * K_BLOCK + i];
            int8_t a_q = static_cast<int8_t>(roundf(__float2int_rn(a_val / a_scale)));
            int8_t w_q = s_weight_q[lane_n * K_BLOCK + i];
            dot_i += static_cast<int32_t>(a_q) * static_cast<int32_t>(w_q);
        }

        sum += a_scale * s_weight_scale[lane_n] * static_cast<float>(dot_i);

        __syncthreads();
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // Strategy dispatch based on batch size
    const int M_THRESHOLD = 16;

    if (M <= M_THRESHOLD) {
        // Simple kernel for small batches (memory-bound)
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x, M);

        gemm_q4_0_fp32_simple<<<grid, block>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    } else {
        // Tiled kernel for large batches (compute-bound)
        dim3 block(32, 4);
        dim3 grid((N + 32 - 1) / 32, (M + 4 - 1) / 4);

        gemm_q4_0_fp32_tiled<<<grid, block>>>(
            w_ptr, act_ptr, out_ptr, M, N, K
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
