#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block structure (llama.cpp compatible)
struct block_q4_0 {
    uint16_t d;      // FP16 scale
    uint8_t qs[16];  // 32 packed quaternions
};

__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    union { uint16_t u16; __half f16; } un;
    un.u16 = h;
    return __half2float(un.f16);
}

// Compute max of 4 floats
__device__ __forceinline__ float fmax4(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

// Optimized kernel: each thread computes one output element
// Uses vectorized loads and __dp4a for efficiency
__global__ void q4_0_gemm_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;
    const block_q4_0* weight_blocks = reinterpret_cast<const block_q4_0*>(weight);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = tid / N;
    const int n = tid % N;

    if (m >= M || n >= N) return;

    const float* a_row = activation + m * K;
    const block_q4_0* w_col = weight_blocks + n * num_blocks;

    // Accumulator
    float sum = 0.0f;

    // Process each block
    for (int b = 0; b < num_blocks; ++b) {
        // Extract weight scale (FP16)
        float d_w = fp16_to_fp32(w_col[b].d);

        // Unpack weights to INT8 for dp4a
        int8_t w_q[32];
        const uint8_t* w_packed = w_col[b].qs;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            w_q[i] = (w_packed[i] & 0x0F) - 8;
            w_q[i + 16] = ((w_packed[i] >> 4) & 0x0F) - 8;
        }

        // Load activation values and compute max for quantization
        float a_vals[32];
        float a_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_vals[i] = a_row[b * 32 + i];
            a_max = fmaxf(a_max, fabsf(a_vals[i]));
        }

        float d_a = a_max / 127.0f;
        if (d_a < 1e-6f) d_a = 1.0f;

        // Quantize activation to INT8
        int8_t a_q[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            a_q[i] = __float2int_rn(a_vals[i] / d_a);
        }

        // Compute dot product using __dp4a (4 MACs per call)
        int sumi = 0;
        const int* a_q_ptr = reinterpret_cast<const int*>(a_q);
        const int* w_q_ptr = reinterpret_cast<const int*>(w_q);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sumi = __dp4a(a_q_ptr[i], w_q_ptr[i], sumi);
        }

        // Apply scaling
        sum += d_w * d_a * sumi;
    }

    output[m * N + n] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const float* a_ptr = activation.data_ptr<float>();
    const uint8_t* w_ptr = weight.data_ptr<uint8_t>();
    float* o_ptr = output.data_ptr<float>();

    // Use 256 threads per block for good occupancy
    int num_threads = 256;
    int num_blocks = (M * N + num_threads - 1) / num_threads;
    q4_0_gemm_kernel<<<num_blocks, num_threads>>>(
        w_ptr, a_ptr, o_ptr, M, N, K
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_0 Quantized GEMM");
}
