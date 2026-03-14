#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Q4_0 format: 18 bytes per block
// 2 bytes: FP16 scale
// 16 bytes: 32 packed 4-bit values (each byte contains 2 values)

// Helper function to unpack 4-bit values from Q4_0 format
// Q4_0 packing: byte[i] = q[i] | (q[i+16] << 4)
__device__ __forceinline__ void unpack_q4_0_block(
    const uint8_t* __restrict__ qs_ptr,
    int* __restrict__ q_values
) {
    // Use scalar loads to avoid alignment issues
    uint8_t b0 = qs_ptr[0], b1 = qs_ptr[1], b2 = qs_ptr[2], b3 = qs_ptr[3];
    uint8_t b4 = qs_ptr[4], b5 = qs_ptr[5], b6 = qs_ptr[6], b7 = qs_ptr[7];
    uint8_t b8 = qs_ptr[8], b9 = qs_ptr[9], b10 = qs_ptr[10], b11 = qs_ptr[11];
    uint8_t b12 = qs_ptr[12], b13 = qs_ptr[13], b14 = qs_ptr[14], b15 = qs_ptr[15];

    // low nibble = q[i], high nibble = q[i+16]
    q_values[0] = b0 & 0x0F; q_values[16] = (b0 >> 4) & 0x0F;
    q_values[1] = b1 & 0x0F; q_values[17] = (b1 >> 4) & 0x0F;
    q_values[2] = b2 & 0x0F; q_values[18] = (b2 >> 4) & 0x0F;
    q_values[3] = b3 & 0x0F; q_values[19] = (b3 >> 4) & 0x0F;
    q_values[4] = b4 & 0x0F; q_values[20] = (b4 >> 4) & 0x0F;
    q_values[5] = b5 & 0x0F; q_values[21] = (b5 >> 4) & 0x0F;
    q_values[6] = b6 & 0x0F; q_values[22] = (b6 >> 4) & 0x0F;
    q_values[7] = b7 & 0x0F; q_values[23] = (b7 >> 4) & 0x0F;
    q_values[8] = b8 & 0x0F; q_values[24] = (b8 >> 4) & 0x0F;
    q_values[9] = b9 & 0x0F; q_values[25] = (b9 >> 4) & 0x0F;
    q_values[10] = b10 & 0x0F; q_values[26] = (b10 >> 4) & 0x0F;
    q_values[11] = b11 & 0x0F; q_values[27] = (b11 >> 4) & 0x0F;
    q_values[12] = b12 & 0x0F; q_values[28] = (b12 >> 4) & 0x0F;
    q_values[13] = b13 & 0x0F; q_values[29] = (b13 >> 4) & 0x0F;
    q_values[14] = b14 & 0x0F; q_values[30] = (b14 >> 4) & 0x0F;
    q_values[15] = b15 & 0x0F; q_values[31] = (b15 >> 4) & 0x0F;
}

// Simple kernel that works correctly (based on v1)
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
        // Load scale (FP16) - use byte-wise access to ensure alignment
        uint8_t scale_bytes[2] = {w_ptr[block * 18], w_ptr[block * 18 + 1]};
        uint16_t scale_u16 = (scale_bytes[1] << 8) | scale_bytes[0];
        half scale_half = *reinterpret_cast<half*>(&scale_u16);
        float d_w = __half2float(scale_half);

        const uint8_t* qs_ptr = w_ptr + block * 18 + 2;
        const float* act_ptr = act_row + block * 32;

        // Unpack 4-bit values and compute
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

// Small batch kernel (M <= 16) - Memory-bound optimization
__global__ void q4_0_fp32_gemm_small_batch_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_blocks = K / 32;
    const int TN = 4;

    const int tid = threadIdx.x;
    const int m_idx = blockIdx.y;
    const int n_base = blockIdx.x * blockDim.x * TN;

    if (m_idx >= M) return;

    const float* act_row = activation + m_idx * K;
    float accum[TN] = {0.0f};

    for (int block = 0; block < K_blocks; block++) {
        const float* act_ptr = act_row + block * 32;
        float act_arr[32];

        for (int i = 0; i < 32; i++) {
            act_arr[i] = act_ptr[i];
        }

        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int n_idx = n_base + tid * TN + tn;
            if (n_idx >= N) continue;

            const uint8_t* w_ptr = weight_q + n_idx * K_blocks * 18 + block * 18;

            // Load scale (FP16) - byte-wise
            uint8_t scale_bytes[2] = {w_ptr[0], w_ptr[1]};
            uint16_t scale_u16 = (scale_bytes[1] << 8) | scale_bytes[0];
            half scale_half = *reinterpret_cast<half*>(&scale_u16);
            float d_w = __half2float(scale_half);

            const uint8_t* qs_ptr = w_ptr + 2;
            int q_values[32];
            unpack_q4_0_block(qs_ptr, q_values);

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int q_low = q_values[i];
                int q_high = q_values[i + 16];
                float w_low = d_w * (float(q_low) - 8.0f);
                float w_high = d_w * (float(q_high) - 8.0f);
                accum[tn] += w_low * act_arr[i];
                accum[tn] += w_high * act_arr[i + 16];
            }
        }
    }

    #pragma unroll
    for (int tn = 0; tn < TN; tn++) {
        int n_idx = n_base + tid * TN + tn;
        if (n_idx < N) {
            output[m_idx * N + n_idx] = accum[tn];
        }
    }
}

// Medium batch kernel (16 < M <= 64) - Balanced optimization
__global__ void q4_0_fp32_gemm_medium_batch_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_blocks = K / 32;
    const int TN = 8;

    int m_idx = blockIdx.x;
    if (m_idx >= M) return;

    int tid = threadIdx.x;
    int n_start = tid * TN;
    const float* act_row = activation + m_idx * K;

    float accum[TN] = {0.0f};

    for (int block = 0; block < K_blocks; block++) {
        const float* act_ptr = act_row + block * 32;

        float a0 = act_ptr[0], a1 = act_ptr[1], a2 = act_ptr[2], a3 = act_ptr[3];
        float a4 = act_ptr[4], a5 = act_ptr[5], a6 = act_ptr[6], a7 = act_ptr[7];
        float a8 = act_ptr[8], a9 = act_ptr[9], a10 = act_ptr[10], a11 = act_ptr[11];
        float a12 = act_ptr[12], a13 = act_ptr[13], a14 = act_ptr[14], a15 = act_ptr[15];
        float a16 = act_ptr[16], a17 = act_ptr[17], a18 = act_ptr[18], a19 = act_ptr[19];
        float a20 = act_ptr[20], a21 = act_ptr[21], a22 = act_ptr[22], a23 = act_ptr[23];
        float a24 = act_ptr[24], a25 = act_ptr[25], a26 = act_ptr[26], a27 = act_ptr[27];
        float a28 = act_ptr[28], a29 = act_ptr[29], a30 = act_ptr[30], a31 = act_ptr[31];

        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int n_idx = n_start + tn;
            if (n_idx >= N) continue;

            const uint8_t* w_block = weight_q + n_idx * K_blocks * 18 + block * 18;

            // Load scale
            uint8_t scale_bytes[2] = {w_block[0], w_block[1]};
            uint16_t scale_u16 = (scale_bytes[1] << 8) | scale_bytes[0];
            half scale_half = *reinterpret_cast<half*>(&scale_u16);
            float d_w = __half2float(scale_half);

            const uint8_t* qs_ptr = w_block + 2;
            int q_values[32];
            unpack_q4_0_block(qs_ptr, q_values);

            // Compute using loop unrolling
            accum[tn] += d_w * ((q_values[0] - 8) * a0 + (q_values[16] - 8) * a16);
            accum[tn] += d_w * ((q_values[1] - 8) * a1 + (q_values[17] - 8) * a17);
            accum[tn] += d_w * ((q_values[2] - 8) * a2 + (q_values[18] - 8) * a18);
            accum[tn] += d_w * ((q_values[3] - 8) * a3 + (q_values[19] - 8) * a19);
            accum[tn] += d_w * ((q_values[4] - 8) * a4 + (q_values[20] - 8) * a20);
            accum[tn] += d_w * ((q_values[5] - 8) * a5 + (q_values[21] - 8) * a21);
            accum[tn] += d_w * ((q_values[6] - 8) * a6 + (q_values[22] - 8) * a22);
            accum[tn] += d_w * ((q_values[7] - 8) * a7 + (q_values[23] - 8) * a23);
            accum[tn] += d_w * ((q_values[8] - 8) * a8 + (q_values[24] - 8) * a24);
            accum[tn] += d_w * ((q_values[9] - 8) * a9 + (q_values[25] - 8) * a25);
            accum[tn] += d_w * ((q_values[10] - 8) * a10 + (q_values[26] - 8) * a26);
            accum[tn] += d_w * ((q_values[11] - 8) * a11 + (q_values[27] - 8) * a27);
            accum[tn] += d_w * ((q_values[12] - 8) * a12 + (q_values[28] - 8) * a28);
            accum[tn] += d_w * ((q_values[13] - 8) * a13 + (q_values[29] - 8) * a29);
            accum[tn] += d_w * ((q_values[14] - 8) * a14 + (q_values[30] - 8) * a30);
            accum[tn] += d_w * ((q_values[15] - 8) * a15 + (q_values[31] - 8) * a31);
        }
    }

    #pragma unroll
    for (int tn = 0; tn < TN; tn++) {
        int n_idx = n_start + tn;
        if (n_idx < N) {
            output[m_idx * N + n_idx] = accum[tn];
        }
    }
}

// Large batch kernel (M > 64) - Compute-bound optimization
__global__ void q4_0_fp32_gemm_large_batch_kernel(
    const uint8_t* __restrict__ weight_q,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int K_blocks = K / 32;
    const int TN = 16;
    const int TM = 1;

    int m_base = blockIdx.y * blockDim.y * TM;
    int n_base = blockIdx.x * blockDim.x * TN;
    int m_idx = m_base + threadIdx.y * TM;
    int tid_x = threadIdx.x;

    if (m_idx >= M) return;

    float accum[TN] = {0.0f};
    const float* act_row = activation + m_idx * K;

    for (int block = 0; block < K_blocks; block++) {
        const float* act_ptr = act_row + block * 32;

        float a[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            a[i] = act_ptr[i];
        }

        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int n_idx = n_base + tid_x * TN + tn;
            if (n_idx >= N) continue;

            const uint8_t* w_block = weight_q + n_idx * K_blocks * 18 + block * 18;

            // Load scale
            uint8_t scale_bytes[2] = {w_block[0], w_block[1]};
            uint16_t scale_u16 = (scale_bytes[1] << 8) | scale_bytes[0];
            half scale_half = *reinterpret_cast<half*>(&scale_u16);
            float d_w = __half2float(scale_half);

            const uint8_t* qs_ptr = w_block + 2;
            int q_values[32];
            unpack_q4_0_block(qs_ptr, q_values);

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                accum[tn] += d_w * ((q_values[i] - 8) * a[i] + (q_values[i + 16] - 8) * a[i + 16]);
            }
        }
    }

    #pragma unroll
    for (int tn = 0; tn < TN; tn++) {
        int n_idx = n_base + tid_x * TN + tn;
        if (n_idx < N) {
            output[m_idx * N + n_idx] = accum[tn];
        }
    }
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

    // Dispatch based on M size for optimal performance
    if (M <= 16) {
        // Small batch: use small batch kernel
        const int TN = 4;
        int threads = min(256, (N + TN - 1) / TN);
        dim3 block(threads);
        dim3 grid(((N + TN - 1) / TN + threads - 1) / threads, M);
        q4_0_fp32_gemm_small_batch_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);
    } else if (M <= 64) {
        // Medium batch: balanced
        const int TN = 8;
        int threads = min(256, (N + TN - 1) / TN);
        dim3 block(threads);
        dim3 grid(M);
        q4_0_fp32_gemm_medium_batch_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);
    } else {
        // Large batch: compute-bound
        const int TN = 16;
        const int TM = 1;
        int threads_x = min(256 / 2, (N + TN - 1) / TN);
        int threads_y = 2;
        dim3 block(threads_x, threads_y);
        dim3 grid(((N + TN - 1) / TN + threads_x - 1) / threads_x, (M + TM - 1) / TM);
        q4_0_fp32_gemm_large_batch_kernel<<<grid, block>>>(weight_q, activation_ptr, output_ptr, M, N, K);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Q4_0 x FP32 GEMM");
}
