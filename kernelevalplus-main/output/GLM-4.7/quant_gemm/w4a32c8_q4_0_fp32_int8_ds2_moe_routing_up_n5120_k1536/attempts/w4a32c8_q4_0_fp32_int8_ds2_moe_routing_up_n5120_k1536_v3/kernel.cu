#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// BLOCK_Q4_0 format: 18 bytes per block
// Optimized kernel with shared memory tiling and better memory access patterns

// Unpack Q4_0 value: returns [0, 15]
__device__ __inline__ int8_t unpack_q4_0(const uint8_t* packed, int idx) {
    if (idx < 16) {
        return (packed[idx] & 0x0F);
    } else {
        return (packed[idx - 16] >> 4) & 0x0F;
    }
}

// Optimized dot product using DP4A instruction (Compute Capability >= 6.1)
// Each DP4A computes: sum += dot(int4_a, int4_b)
__device__ __inline__ void vec_dot_q4_0_fp32_dp4a(
    const uint8_t* __restrict__ weight_block,  // 18 bytes
    const float* __restrict__ act_block,        // 32 floats
    float* __restrict__ a_sum_out,              // Output: sum of activations
    int32_t* __restrict__ sumi_out              // Output: sum of q_w * q_a
) {
    const uint8_t* packed_qs = weight_block + 2;
    half d_w_half = *((const half*)weight_block);
    float d_w = __half2float(d_w_half);

    // Compute activation stats
    float a_max = 0.0f;
    float a_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float a_val = act_block[i];
        a_max = fmaxf(a_max, fabsf(a_val));
        a_sum += a_val;
    }
    *a_sum_out = a_sum;

    float d_a = a_max / 127.0f;
    if (d_a < 1e-7f) d_a = 1e-7f;

    // Quantize activation to int8
    int8_t a_q[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        a_q[i] = __float2int_rn(act_block[i] / d_a);
    }

    // Unpack Q4_0 weights to int32 (4 packed values per int32 for DP4A)
    // Q4_0: byte[i] = q[i] | (q[i+16] << 4)
    // We need to unpack and offset by 8
    int32_t w_packed[8];  // 8 int32s = 32 Q4 values

    // Unpack 16 Q4_0 bytes to 32 int8 values in [-8, 7]
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t byte_val = packed_qs[i];
        int8_t low = (byte_val & 0x0F) - 8;      // [-8, 7]
        int8_t high = ((byte_val >> 4) & 0x0F) - 8; // [-8, 7]
        // Pack two int8s into one int16
        ((int16_t*)w_packed)[i] = (int16_t)((int16_t)high << 8) | (uint16_t)(uint8_t)low;
    }

    // Use DP4A for 4 pairs at a time
    int32_t sumi = 0;

    // Process with DP4A (4 pairs per instruction)
    int32_t* a_packed = (int32_t*)a_q;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;\n"
            : "+r"(sumi)
            : "r"(w_packed[i]), "r"(a_packed[i]), "r"(sumi));
    }

    *sumi_out = sumi;
}

// Kernel with 2D thread block for better occupancy
// Each thread block computes a tile of output
__global__ void quant_gemm_q4_0_fp32_kernel_v2(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int TILE_M = 4;   // Tile size in M dimension
    const int TILE_N = 32;  // Tile size in N dimension (one Q4_0 block)

    // Thread ID mapping
    const int tid_m = threadIdx.x / 32;  // 0-3
    const int tid_n = threadIdx.x % 32;  // 0-31

    const int m_base = blockIdx.x * TILE_M + tid_m;
    const int n_base = blockIdx.y * TILE_N;

    if (m_base >= M || n_base >= N) return;

    const int n_end = min(n_base + TILE_N, N);
    const int k_blocks = K / 32;

    // Accumulator for this thread's output element
    float acc = 0.0f;

    // Shared memory for activation block
    __shared__ float s_act[4][32];  // TILE_M x 32

    // Process each K block
    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * 32;

        // Load activation block cooperatively
        #pragma unroll
        for (int i = 0; i < TILE_M; i++) {
            int m_load = blockIdx.x * TILE_M + i;
            if (m_load < M && threadIdx.x < 32) {
                s_act[i][threadIdx.x] = activation[m_load * K + k_base + threadIdx.x];
            }
        }
        __syncthreads();

        // Each thread computes one output element
        if (tid_n < (n_end - n_base)) {
            const int n_cur = n_base + tid_n;
            const int weight_block_idx = n_cur * k_blocks + kb;
            const uint8_t* weight_block = (const uint8_t*)weight + weight_block_idx * 18;

            float a_sum;
            int32_t sumi;
            vec_dot_q4_0_fp32_dp4a(weight_block, s_act[tid_m], &a_sum, &sumi);

            half d_w_half = *((const half*)weight_block);
            float d_w = __half2float(d_w_half);

            // Compute activation scale for this block
            float a_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                a_max = fmaxf(a_max, fabsf(s_act[tid_m][i]));
            }
            float d_a = a_max / 127.0f;
            if (d_a < 1e-7f) d_a = 1e-7f;

            acc += d_w * (d_a * (float)sumi - 8.0f * a_sum);
        }
        __syncthreads();
    }

    // Write result
    if (tid_n < (n_end - n_base)) {
        output[m_base * N + n_base + tid_n] = acc;
    }
}

// Simplified high-performance kernel
// Optimized for N=5120, K=1536 (48 K-blocks of 32)
__global__ void quant_gemm_q4_0_fp32_kernel_v3(
    const void* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int m_idx = blockIdx.x;
    if (m_idx >= M) return;

    const int k_blocks = K / 32;  // 48 for K=1536
    const int n_per_thread = 32;  // Each thread computes 32 output values

    const int n_base = blockIdx.y * blockDim.x * n_per_thread + threadIdx.x * n_per_thread;
    if (n_base >= N) return;

    const int n_end = min(n_base + n_per_thread, N);
    const int n_valid = n_end - n_base;

    // Accumulators
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }

    // Process K blocks
    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * 32;

        // Load activation block (coalesced read)
        float act_block[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            act_block[i] = activation[m_idx * K + k_base + i];
        }

        // Process each N in this thread's chunk
        for (int nb = 0; nb < n_valid; nb++) {
            const int n_cur = n_base + nb;
            const int weight_block_idx = n_cur * k_blocks + kb;
            const uint8_t* weight_block = (const uint8_t*)weight + weight_block_idx * 18;

            // Read scale
            half d_w_half = *((const half*)weight_block);
            float d_w = __half2float(d_w_half);
            const uint8_t* packed_qs = weight_block + 2;

            // Activation stats
            float a_max = 0.0f;
            float a_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float a_val = act_block[i];
                a_max = fmaxf(a_max, fabsf(a_val));
                a_sum += a_val;
            }
            float d_a = a_max / 127.0f;
            if (d_a < 1e-7f) d_a = 1e-7f;

            // Compute sumi
            int32_t sumi = 0;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int8_t q_w = unpack_q4_0(packed_qs, i);
                int8_t q_a = __float2int_rn(act_block[i] / d_a);
                sumi += q_w * q_a;
            }

            acc[nb] += d_w * (d_a * (float)sumi - 8.0f * a_sum);
        }
    }

    // Write results
    for (int nb = 0; nb < n_valid; nb++) {
        output[m_idx * N + n_base + nb] = acc[nb];
    }
}

// Host wrapper
torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat32).device(activation.device()));

    const int k_blocks = K / 32;
    const int n_blocks = (N + 31) / 32;

    // Choose configuration based on M
    if (M == 1) {
        // Single batch: maximize parallelism across N
        const int threads_per_block = 256;
        const int n_chunks = (n_blocks + 7) / 8;  // Each thread handles one block

        dim3 grid(M, n_chunks);
        dim3 block(threads_per_block);

        quant_gemm_q4_0_fp32_kernel_v3<<<grid, block>>>(
            weight.data_ptr(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else if (M <= 16) {
        // Small batch: balance between M and N parallelism
        const int threads_per_block = 128;

        dim3 grid(M, n_blocks);
        dim3 block(threads_per_block);

        quant_gemm_q4_0_fp32_kernel_v3<<<grid, block>>>(
            weight.data_ptr(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Large batch: process M with good occupancy
        const int threads_per_block = 256;
        const int n_chunks = (n_blocks * 32 + threads_per_block - 1) / threads_per_block;

        dim3 grid(M, n_chunks);
        dim3 block(threads_per_block);

        quant_gemm_q4_0_fp32_kernel_v3<<<grid, block>>>(
            weight.data_ptr(),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Quantized GEMM (W4A32C8, Q4_0 weights, FP32 activation)");
}
