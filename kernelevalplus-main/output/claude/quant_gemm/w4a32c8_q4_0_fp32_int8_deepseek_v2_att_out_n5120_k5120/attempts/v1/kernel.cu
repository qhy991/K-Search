#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 format: 18 bytes per block
// 2 bytes: FP16 scale
// 16 bytes: 32 packed 4-bit values (each byte contains 2 values)

// Small batch kernel (M <= 8) - Memory-bound optimization
// Uses vectorized loads, simple structure to maximize bandwidth
__global__ void q4_0_fp32_gemm_small_batch_kernel(
    const uint8_t* __restrict__ weight_q,     // [N, K/32] in Q4_0 format
    const float* __restrict__ activation,     // [M, K] FP32
    float* __restrict__ output,               // [M, N] FP32
    int M, int N, int K
) {
    const int K_blocks = K / 32;

    // Each thread handles one output element
    const int m_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    // Load activation row once per M
    const float* act_row = activation + m_idx * K;

    // Pointer to weight for this N
    const uint8_t* w_ptr = weight_q + n_idx * K_blocks * 18;

    // Accumulate result
    float sum = 0.0f;

    // Process K dimension in blocks of 32
    for (int block = 0; block < K_blocks; block++) {
        // Load weight block scale (FP16) - use scalar load to avoid alignment issues
        half2 scale_data = *reinterpret_cast<const half2*>(w_ptr + block * 18);
        float d_w = __half2float(scale_data.x);

        // Process 32 elements
        const uint8_t* qs_ptr = w_ptr + block * 18 + 2;
        const float* act_ptr = act_row + block * 32;

        // Use aligned vectorized loads for activation (each float is 4 bytes)
        float act_arr[32];

        // Load 32 floats using vectorized loads (4 floats per float4 = 16 bytes)
        // Load in chunks of 8 floats (32 bytes each) for alignment
        float4 a0 = *reinterpret_cast<const float4*>(act_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(act_ptr + 4);
        float4 a2 = *reinterpret_cast<const float4*>(act_ptr + 8);
        float4 a3 = *reinterpret_cast<const float4*>(act_ptr + 12);
        float4 a4 = *reinterpret_cast<const float4*>(act_ptr + 16);
        float4 a5 = *reinterpret_cast<const float4*>(act_ptr + 20);
        float4 a6 = *reinterpret_cast<const float4*>(act_ptr + 24);
        float4 a7 = *reinterpret_cast<const float4*>(act_ptr + 28);

        // Extract floats from float4
        act_arr[0] = a0.x; act_arr[1] = a0.y; act_arr[2] = a0.z; act_arr[3] = a0.w;
        act_arr[4] = a1.x; act_arr[5] = a1.y; act_arr[6] = a1.z; act_arr[7] = a1.w;
        act_arr[8] = a2.x; act_arr[9] = a2.y; act_arr[10] = a2.z; act_arr[11] = a2.w;
        act_arr[12] = a3.x; act_arr[13] = a3.y; act_arr[14] = a3.z; act_arr[15] = a3.w;
        act_arr[16] = a4.x; act_arr[17] = a4.y; act_arr[18] = a4.z; act_arr[19] = a4.w;
        act_arr[20] = a5.x; act_arr[21] = a5.y; act_arr[22] = a5.z; act_arr[23] = a5.w;
        act_arr[24] = a6.x; act_arr[25] = a6.y; act_arr[26] = a6.z; act_arr[27] = a6.w;
        act_arr[28] = a7.x; act_arr[29] = a7.y; act_arr[30] = a7.z; act_arr[31] = a7.w;

        // Load packed 4-bit values (16 bytes) using aligned load
        uint4 qs_vec = *reinterpret_cast<const uint4*>(qs_ptr);

        // Unpack and compute dot product
        // Q4_0 encoding: q = round(val/scale + 8), decode: val = scale * (q - 8)
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            // Extract byte from uint4 (little endian)
            uint32_t qs_u32 = qs_vec.x;
            if (i >= 4) qs_u32 = qs_vec.y;
            if (i >= 8) qs_u32 = qs_vec.z;
            if (i >= 12) qs_u32 = qs_vec.w;

            int byte_idx = i % 4;
            uint8_t byte = (qs_u32 >> (byte_idx * 8)) & 0xFF;
            int q_low = byte & 0x0F;
            int q_high = (byte >> 4) & 0x0F;

            float val_low = d_w * (float(q_low) - 8.0f);
            float val_high = d_w * (float(q_high) - 8.0f);

            sum += val_low * act_arr[i];
            sum += val_high * act_arr[i + 16];
        }
    }

    output[m_idx * N + n_idx] = sum;
}

// Simplified batch kernel - better for correctness
__global__ void q4_0_fp32_gemm_simple_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_blocks = K / 32;

    // Each thread handles one output element
    const int m_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    const float* act_row = activation + m_idx * K;
    const uint8_t* w_ptr = weight_q + n_idx * K_blocks * 18;

    float sum = 0.0f;

    for (int block = 0; block < K_blocks; block++) {
        // Load scale (FP16) as half2 to ensure alignment
        half2 scale_data = *reinterpret_cast<const half2*>(w_ptr + block * 18);
        float d_w = __half2float(scale_data.x);

        const uint8_t* qs_ptr = w_ptr + block * 18 + 2;
        const float* act_ptr = act_row + block * 32;

        // Unpack 4-bit values and compute
        // Q4_0 format: 16 bytes containing 32 packed 4-bit values
        // Packing: byte[i] contains q[i] in low nibble, q[i+16] in high nibble

        for (int i = 0; i < 16; i++) {
            uint8_t byte = qs_ptr[i];
            int q_low = byte & 0x0F;
            int q_high = (byte >> 4) & 0x0F;

            float w_low = d_w * (float(q_low) - 8.0f);
            float w_high = d_w * (float(q_high) - 8.0f);

            sum += w_low * act_ptr[i];
            sum += w_high * act_ptr[i + 16];
        }
    }

    output[m_idx * N + n_idx] = sum;
}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const uint8_t* weight_q = weight.data_ptr<uint8_t>();
    const float* activation_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Use simple kernel for all M sizes for correctness first
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    q4_0_fp32_gemm_simple_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
