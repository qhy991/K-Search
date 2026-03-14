#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define QK 32
#define WARP_SIZE 32
#define BLOCK_Q4_0_SIZE 18

// Device function to read FP16 as float
__device__ __inline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

/**
 * W4A32C8 Q4_0 Quantized GEMM Kernel v8 - Optimized for all M
 *
 * Key improvements from v7:
 * 1. Shared memory for activation quantization in all kernels
 * 2. Better thread block organization for large M
 * 3. Reduced redundant quantization computation
 */
__global__ void __launch_bounds__(1024) gemm_q4_0_v8_shared(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    extern __shared__ char smem_raw[];
    float* s_d_a = reinterpret_cast<float*>(smem_raw);
    float* s_s_a = s_d_a + (K / QK);
    int8_t* s_a_qs = reinterpret_cast<int8_t*>(s_s_a + (K / QK));

    const int num_blocks_k = K / QK;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Each block processes multiple M rows
    const int m_base = blockIdx.y * blockDim.y + threadIdx.y;

    if (m_base >= M) return;

    // Phase 1: Cooperatively compute and cache activation quantization
    for (int kb = tid; kb < num_blocks_k; kb += blockDim.x) {
        const float* act_ptr = activation + m_base * K + kb * QK;

        float a[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a[i] = act_ptr[i];
        }

        float a_max = 0.0f, a_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a_max = fmaxf(a_max, fabsf(a[i]));
            a_sum += a[i];
        }

        const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
        const float inv_d_a = 1.0f / d_a;

        s_d_a[kb] = d_a;
        s_s_a[kb] = a_sum;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            s_a_qs[kb * 32 + i] = (int8_t)__float2int_rn(a[i] * inv_d_a);
        }
    }

    __syncthreads();

    // Phase 2: Compute outputs
    const int n = blockIdx.x * num_warps + warp_id;

    if (n >= N) return;

    const uint8_t* w_row = weight + (long long)n * num_blocks_k * BLOCK_Q4_0_SIZE;

    float sum = 0.0f;

    for (int kb = lane_id; kb < num_blocks_k; kb += WARP_SIZE) {
        const uint8_t* block_ptr = w_row + kb * BLOCK_Q4_0_SIZE;

        uint16_t d_raw = block_ptr[0] | (block_ptr[1] << 8);
        const float d_w = read_half_as_float(d_raw);

        uint8_t qs[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            qs[i] = block_ptr[2 + i];
        }

        const float d_a = s_d_a[kb];
        const float s_a = s_s_a[kb];
        const int8_t* a_qs = &s_a_qs[kb * 32];

        int a_packed[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a_packed[i] = *reinterpret_cast<const int*>(&a_qs[i * 4]);
        }

        int sumi = 0;
        uint32_t wp[4];
        for (int i = 0; i < 4; i++) {
            wp[i] = (uint32_t)qs[i * 4 + 0] |
                   ((uint32_t)qs[i * 4 + 1] << 8) |
                   ((uint32_t)qs[i * 4 + 2] << 16) |
                   ((uint32_t)qs[i * 4 + 3] << 24);
        }

        asm volatile(
            "dp4a.u32.s32 %0, %1, %2, %0;\n\t"
            "dp4a.u32.s32 %0, %3, %4, %0;\n\t"
            "dp4a.u32.s32 %0, %5, %6, %0;\n\t"
            "dp4a.u32.s32 %0, %7, %8, %0;\n\t"
            "dp4a.u32.s32 %0, %9, %10, %0;\n\t"
            "dp4a.u32.s32 %0, %11, %12, %0;\n\t"
            "dp4a.u32.s32 %0, %13, %14, %0;\n\t"
            "dp4a.u32.s32 %0, %15, %16, %0;\n\t"
            : "+r"(sumi)
            : "r"(wp[0] & 0x0F0F0F0F), "r"(a_packed[0]),
              "r"((wp[0] >> 4) & 0x0F0F0F0F), "r"(a_packed[4]),
              "r"(wp[1] & 0x0F0F0F0F), "r"(a_packed[1]),
              "r"((wp[1] >> 4) & 0x0F0F0F0F), "r"(a_packed[5]),
              "r"(wp[2] & 0x0F0F0F0F), "r"(a_packed[2]),
              "r"((wp[2] >> 4) & 0x0F0F0F0F), "r"(a_packed[6]),
              "r"(wp[3] & 0x0F0F0F0F), "r"(a_packed[3]),
              "r"((wp[3] >> 4) & 0x0F0F0F0F), "r"(a_packed[7])
        );

        sum += d_w * (d_a * (float)sumi - 8.0f * s_a);
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        output[m_base * N + n] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kByte, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    const int num_blocks_k = K / QK;
    const int threads_per_block = 1024;
    const int warps_per_block = threads_per_block / WARP_SIZE;
    const int blocks_x = (N + warps_per_block - 1) / warps_per_block;
    const int blocks_y = M;

    const size_t smem_size = num_blocks_k * (sizeof(float) + sizeof(float) + 32);

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block, 1);

    gemm_q4_0_v8_shared<<<grid, block, smem_size>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM v8 (DeepSeek-V3 LM Head)");
}
