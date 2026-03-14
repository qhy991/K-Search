#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Q4_1 × Q8_1 Quantized GEMM for Qwen3-4B Attention Output Projection
// Task: w4a32c8_q4_1_fp32_int8_qwen3_4b_att_out_n2560_k2560
// v12: Using byte-based weight indexing like ffn_down
// ============================================================================

#define QK4_1 32
#define QK8_1 32
#define WARP_SIZE 32

typedef struct {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
} block_q4_1;

static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size must be 20 bytes");

__device__ __forceinline__ float read_half_as_float(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
           a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(256) gemm_q4_1_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    const int num_blocks = K / QK4_1;

    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int m = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        const int blocks_per_warp = (num_blocks + WARP_SIZE - 1) / WARP_SIZE;

        for (int b_offset = 0; b_offset < blocks_per_warp; b_offset++) {
            const int b = b_offset * WARP_SIZE + lane_id;
            if (b >= num_blocks) continue;

            // Byte-based weight indexing like ffn_down
            const block_q4_1* wb = reinterpret_cast<const block_q4_1*>(
                weight + (n * num_blocks + b) * sizeof(block_q4_1)
            );

            const float d_w = read_half_as_float(wb->d);
            const float m_w = read_half_as_float(wb->m);
            const int k_start = b * QK4_1;

            float a_block[QK8_1];
            const float* act_ptr = &activation[m * K + k_start];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 a4 = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
                a_block[i * 4 + 0] = a4.x;
                a_block[i * 4 + 1] = a4.y;
                a_block[i * 4 + 2] = a4.z;
                a_block[i * 4 + 3] = a4.w;
            }

            float a_max = 0.0f;
            float a_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK8_1; i++) {
                a_max = fmaxf(a_max, fabsf(a_block[i]));
                a_sum += a_block[i];
            }
            const float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
            const float s_a = a_sum;

            int32_t a_packed[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int8_t q0 = (int8_t)__float2int_rn(a_block[i * 4] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(a_block[i * 4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(a_block[i * 4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(a_block[i * 4 + 3] / d_a);
                a_packed[i] = (int((uint8_t)q0)) |
                              (int((uint8_t)q1) << 8) |
                              (int((uint8_t)q2) << 16) |
                              (int((uint8_t)q3) << 24);
            }

            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&wb->qs[i * 4]);
                int8_t w0 = (int8_t)(w_raw & 0x0F);
                int8_t w1 = (int8_t)((w_raw >> 8) & 0x0F);
                int8_t w2 = (int8_t)((w_raw >> 16) & 0x0F);
                int8_t w3 = (int8_t)((w_raw >> 24) & 0x0F);
                int w_pack = (int((uint8_t)w0)) |
                             (int((uint8_t)w1) << 8) |
                             (int((uint8_t)w2) << 16) |
                             (int((uint8_t)w3) << 24);
                sumi = dp4a(a_packed[i], w_pack, sumi);
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t w_raw = *reinterpret_cast<const uint32_t*>(&wb->qs[i * 4]);
                int8_t w0 = (int8_t)((w_raw >> 4) & 0x0F);
                int8_t w1 = (int8_t)((w_raw >> 12) & 0x0F);
                int8_t w2 = (int8_t)((w_raw >> 20) & 0x0F);
                int8_t w3 = (int8_t)((w_raw >> 28) & 0x0F);
                int w_pack = (int((uint8_t)w0)) |
                             (int((uint8_t)w1) << 8) |
                             (int((uint8_t)w2) << 16) |
                             (int((uint8_t)w3) << 24);
                sumi = dp4a(a_packed[i + 4], w_pack, sumi);
            }

            sum += d_w * d_a * (float)sumi + m_w * s_a;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            output[m * N + n] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    AT_ASSERTM(activation.dim() == 2);
    AT_ASSERTM(activation.size(0) == M);
    AT_ASSERTM(activation.size(1) == K);

    const int num_blocks_k = K / 32;
    const int bytes_per_block = 20;

    torch::Tensor weight_contiguous;
    if (weight.dim() == 1) {
        int64_t expected_size = N * num_blocks_k * bytes_per_block;
        AT_ASSERTM(weight.size(0) == expected_size);
        weight_contiguous = weight.contiguous();
    } else if (weight.dim() == 2) {
        AT_ASSERTM(weight.size(0) == N);
        AT_ASSERTM(weight.size(1) == num_blocks_k * bytes_per_block);
        weight_contiguous = weight.contiguous().view({-1});
    } else if (weight.dim() == 3) {
        AT_ASSERTM(weight.size(0) == N);
        AT_ASSERTM(weight.size(1) == num_blocks_k);
        AT_ASSERTM(weight.size(2) == bytes_per_block);
        weight_contiguous = weight.contiguous().view({-1});
    } else {
        AT_ASSERTM(false);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    const uint8_t* weight_ptr = weight_contiguous.data_ptr<uint8_t>();
    const float* act_ptr = activation.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int threads_per_block = 256;
    const int total_output_elements = M * N;
    const int num_warps_per_block = threads_per_block / WARP_SIZE;

    int num_blocks;
    if (M <= 8) {
        num_blocks = min(512, max(384, (total_output_elements + num_warps_per_block - 1) / num_warps_per_block));
    } else {
        num_blocks = min(128, (total_output_elements + num_warps_per_block - 1) / num_warps_per_block);
    }

    dim3 grid(num_blocks);
    dim3 block(threads_per_block);

    gemm_q4_1_kernel<<<grid, block, 0>>>(weight_ptr, act_ptr, out_ptr, M, N, K);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_1 × Q8_1 Quantized GEMM");
}
