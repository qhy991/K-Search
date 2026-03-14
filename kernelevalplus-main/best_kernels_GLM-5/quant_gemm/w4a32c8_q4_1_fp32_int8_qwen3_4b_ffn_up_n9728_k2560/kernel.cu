/**
 * W4A32C8 Quantized GEMM for Qwen3-4B FFN Up/Gate Projection
 * Q4_1 Weight (N=9728, K=2560) x FP32 Activation (M=batch, K=2560)
 *
 * Q4_1 format (20 bytes): d (FP16) + m (FP16) + qs[16] (uint8)
 * Q4_1 formula: result = d_w * d_a * sumi + m_w * a_sum
 *
 * v6: Shared memory activation caching for all M <= 8
 * Each block handles one M row, caching quantized activation
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define QK4_1 32
#define WARP_SIZE 32
#define Q4_1_BLOCK 20
#define NUM_K_BLOCKS 80  // 2560 / 32

inline __device__ float read_half_as_float(uint16_t h) {
    half hf;
    memcpy(&hf, &h, sizeof(uint16_t));
    return __half2float(hf);
}

#if __CUDA_ARCH__ >= 610
inline __device__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
inline __device__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Shared memory structure for quantized activation blocks
struct quant_block {
    float scale;    // d_a
    float sum;      // a_sum
    int8_t qs[QK4_1];
};

/**
 * M=1 kernel with shared memory activation caching
 */
__global__ void __launch_bounds__(256) gemm_m1_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int N)
{
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / WARP_SIZE;

    __shared__ quant_block act_blocks[NUM_K_BLOCKS];

    // Phase 1: Cooperatively quantize activation into shared memory
    for (int kb = tid; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_start = kb * QK4_1;

        float a_vals[QK4_1];
        #pragma unroll
        for (int i = 0; i < QK4_1; i += 4) {
            const float4* ptr4 = reinterpret_cast<const float4*>(&activation[k_start + i]);
            float4 v = *ptr4;
            a_vals[i] = v.x; a_vals[i+1] = v.y; a_vals[i+2] = v.z; a_vals[i+3] = v.w;
        }

        float a_max = 0.0f;
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
            a_sum += a_vals[i];
        }

        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float scale = 127.0f / fmaxf(a_max, 1e-10f);

        act_blocks[kb].scale = d_a;
        act_blocks[kb].sum = a_sum;

        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            act_blocks[kb].qs[i] = (int8_t)__float2int_rn(a_vals[i] * scale);
        }
    }
    __syncthreads();

    // Phase 2: Each warp computes multiple outputs
    const int outputs_per_warp = 4;
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    const int total_warps = gridDim.x * (blockDim.x / WARP_SIZE);

    for (int n_base = global_warp_id * outputs_per_warp; n_base < N; n_base += total_warps * outputs_per_warp) {
        float partial_sums[4] = {0.0f};
        int n_vals[4];
        bool valid[4];

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            n_vals[o] = n_base + o;
            valid[o] = (n_vals[o] < N);
        }

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const float d_a = act_blocks[kb].scale;
            const float a_sum = act_blocks[kb].sum;
            const int8_t* a_qs = act_blocks[kb].qs;

            #pragma unroll
            for (int o = 0; o < 4; o++) {
                if (!valid[o]) continue;

                const uint8_t* w_block = weight + (static_cast<int64_t>(n_vals[o]) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
                const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
                const float m_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
                const uint8_t* qs = w_block + 4;

                int32_t sumi = 0;

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint8_t b0 = qs[i * 4 + 0];
                    uint8_t b1 = qs[i * 4 + 1];
                    uint8_t b2 = qs[i * 4 + 2];
                    uint8_t b3 = qs[i * 4 + 3];
                    int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                                (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);

                    int a_pack = (int((uint8_t)a_qs[i*4])) |
                                (int((uint8_t)a_qs[i*4+1]) << 8) |
                                (int((uint8_t)a_qs[i*4+2]) << 16) |
                                (int((uint8_t)a_qs[i*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint8_t b0 = qs[i * 4 + 0];
                    uint8_t b1 = qs[i * 4 + 1];
                    uint8_t b2 = qs[i * 4 + 2];
                    uint8_t b3 = qs[i * 4 + 3];
                    int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                                (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);

                    int a_pack = (int((uint8_t)a_qs[(i+4)*4])) |
                                (int((uint8_t)a_qs[(i+4)*4+1]) << 8) |
                                (int((uint8_t)a_qs[(i+4)*4+2]) << 16) |
                                (int((uint8_t)a_qs[(i+4)*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                partial_sums[o] += d_w * d_a * (float)sumi + m_w * a_sum;
            }
        }

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            if (!valid[o]) continue;
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                output[n_vals[o]] = partial_sums[o];
            }
        }
    }
}

/**
 * Small M kernel (M <= 8): Shared memory per row, 256 threads per row
 */
__global__ void __launch_bounds__(256) gemm_small_m_shared_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K)
{
    const int row = blockIdx.y;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / WARP_SIZE;

    __shared__ quant_block act_blocks[NUM_K_BLOCKS];

    // Phase 1: Cooperatively quantize activation into shared memory
    const float* row_activation = activation + static_cast<int64_t>(row) * K;

    for (int kb = tid; kb < NUM_K_BLOCKS; kb += blockDim.x) {
        const int k_start = kb * QK4_1;

        float a_vals[QK4_1];
        #pragma unroll
        for (int i = 0; i < QK4_1; i += 4) {
            const float4* ptr4 = reinterpret_cast<const float4*>(&row_activation[k_start + i]);
            float4 v = *ptr4;
            a_vals[i] = v.x; a_vals[i+1] = v.y; a_vals[i+2] = v.z; a_vals[i+3] = v.w;
        }

        float a_max = 0.0f;
        float a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
            a_sum += a_vals[i];
        }

        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float scale = 127.0f / fmaxf(a_max, 1e-10f);

        act_blocks[kb].scale = d_a;
        act_blocks[kb].sum = a_sum;

        #pragma unroll
        for (int i = 0; i < QK4_1; i++) {
            act_blocks[kb].qs[i] = (int8_t)__float2int_rn(a_vals[i] * scale);
        }
    }
    __syncthreads();

    // Phase 2: Each warp computes multiple outputs
    const int outputs_per_warp = 4;
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    const int total_warps = gridDim.x * (blockDim.x / WARP_SIZE);

    for (int n_base = global_warp_id * outputs_per_warp; n_base < N; n_base += total_warps * outputs_per_warp) {
        float partial_sums[4] = {0.0f};
        int n_vals[4];
        bool valid[4];

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            n_vals[o] = n_base + o;
            valid[o] = (n_vals[o] < N);
        }

        for (int kb = lane_id; kb < NUM_K_BLOCKS; kb += WARP_SIZE) {
            const float d_a = act_blocks[kb].scale;
            const float a_sum = act_blocks[kb].sum;
            const int8_t* a_qs = act_blocks[kb].qs;

            #pragma unroll
            for (int o = 0; o < 4; o++) {
                if (!valid[o]) continue;

                const uint8_t* w_block = weight + (static_cast<int64_t>(n_vals[o]) * NUM_K_BLOCKS + kb) * Q4_1_BLOCK;
                const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
                const float m_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
                const uint8_t* qs = w_block + 4;

                int32_t sumi = 0;

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint8_t b0 = qs[i * 4 + 0];
                    uint8_t b1 = qs[i * 4 + 1];
                    uint8_t b2 = qs[i * 4 + 2];
                    uint8_t b3 = qs[i * 4 + 3];
                    int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                                (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);

                    int a_pack = (int((uint8_t)a_qs[i*4])) |
                                (int((uint8_t)a_qs[i*4+1]) << 8) |
                                (int((uint8_t)a_qs[i*4+2]) << 16) |
                                (int((uint8_t)a_qs[i*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint8_t b0 = qs[i * 4 + 0];
                    uint8_t b1 = qs[i * 4 + 1];
                    uint8_t b2 = qs[i * 4 + 2];
                    uint8_t b3 = qs[i * 4 + 3];
                    int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                                (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);

                    int a_pack = (int((uint8_t)a_qs[(i+4)*4])) |
                                (int((uint8_t)a_qs[(i+4)*4+1]) << 8) |
                                (int((uint8_t)a_qs[(i+4)*4+2]) << 16) |
                                (int((uint8_t)a_qs[(i+4)*4+3]) << 24);

                    sumi = dp4a(a_pack, w_pack, sumi);
                }

                partial_sums[o] += d_w * d_a * (float)sumi + m_w * a_sum;
            }
        }

        #pragma unroll
        for (int o = 0; o < 4; o++) {
            if (!valid[o]) continue;
            partial_sums[o] = warp_reduce_sum(partial_sums[o]);
            if (lane_id == 0) {
                output[static_cast<int64_t>(row) * N + n_vals[o]] = partial_sums[o];
            }
        }
    }
}

/**
 * Large M kernel: 256 threads, 64 columns per block
 */
__global__ void __launch_bounds__(256)
gemm_large_m_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int COLS_PER_WARP = 8;
    const int base_col = blockIdx.y * 8 * COLS_PER_WARP + warp_id * COLS_PER_WARP;
    const int cols_to_process = min(COLS_PER_WARP, N - base_col);

    if (base_col >= N) return;

    const int num_k_blocks = K / QK4_1;
    float sums[8] = {0.0f};

    for (int kb = lane_id; kb < num_k_blocks; kb += WARP_SIZE) {
        const int k_start = kb * QK4_1;

        float a_block[32];
        const float* act_ptr = &activation[static_cast<int64_t>(row) * K + k_start];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
            a_block[i * 4 + 0] = a4.x;
            a_block[i * 4 + 1] = a4.y;
            a_block[i * 4 + 2] = a4.z;
            a_block[i * 4 + 3] = a4.w;
        }

        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_max = fmaxf(a_max, fabsf(a_block[i]));
            a_sum += a_block[i];
        }
        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;

        int32_t a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4 + 0] / d_a);
            const int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
            const int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
            const int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
            a_packed[i] = (int(q0) & 0xFF) | ((int(q1) & 0xFF) << 8) |
                          ((int(q2) & 0xFF) << 16) | ((int(q3) & 0xFF) << 24);
        }

        for (int c = 0; c < cols_to_process; ++c) {
            const int col = base_col + c;
            const uint8_t* w_block = weight + (static_cast<int64_t>(col) * num_k_blocks + kb) * Q4_1_BLOCK;
            const float d_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block));
            const float m_w = read_half_as_float(*reinterpret_cast<const uint16_t*>(w_block + 2));
            const uint8_t* qs = w_block + 4;

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0];
                uint8_t b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2];
                uint8_t b3 = qs[i * 4 + 3];
                int w_pack = (int(b0 & 0x0F)) | (int(b1 & 0x0F) << 8) |
                            (int(b2 & 0x0F) << 16) | (int(b3 & 0x0F) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t b0 = qs[i * 4 + 0];
                uint8_t b1 = qs[i * 4 + 1];
                uint8_t b2 = qs[i * 4 + 2];
                uint8_t b3 = qs[i * 4 + 3];
                int w_pack = (int((b0 >> 4) & 0x0F)) | (int((b1 >> 4) & 0x0F) << 8) |
                            (int((b2 >> 4) & 0x0F) << 16) | (int((b3 >> 4) & 0x0F) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sums[c] += d_w * d_a * (float)sumi + m_w * a_sum;
        }
    }

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[c] += __shfl_down_sync(0xffffffff, sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < cols_to_process; ++c) {
            output[static_cast<int64_t>(row) * N + base_col + c] = sums[c];
        }
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(activation.is_contiguous(), "Activation must be contiguous");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    const float* a_ptr = activation.data_ptr<float>();
    float* c_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(weight.device().index());

    if (M == 1) {
        // M=1: Use dedicated single-row kernel with 512 blocks
        const int blocks = 512;
        const int threads = 256;
        gemm_m1_shared_kernel<<<blocks, threads, 0, stream>>>(w_ptr, a_ptr, c_ptr, N);
    } else if (M <= 8) {
        // M<=8: Use shared memory kernel, one block per row
        const int blocks_per_row = 512;
        dim3 grid(blocks_per_row, M);
        dim3 block(256);
        gemm_small_m_shared_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    } else {
        // M>8: Standard large M kernel
        dim3 grid(M, (N + 63) / 64);
        dim3 block(256);
        gemm_large_m_kernel<<<grid, block, 0, stream>>>(w_ptr, a_ptr, c_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_1 Quantized GEMM - Qwen3-4B FFN Up v6 - Shared Memory for Small M");
}
