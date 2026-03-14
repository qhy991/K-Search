#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define QK4_1 32
#define QK8_1 32
#define WARP_SIZE 32

// Optimized tiling parameters for LM Head (large N=102400, small M)
#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define THREADS_M 8
#define THREADS_N 32

#define BATCH_THRESHOLD 16

// Q4_1 block structure (20 bytes per block)
struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

inline __device__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<half*>(&h));
}

#if __CUDA_ARCH__ >= 610
inline __device__ int dp4a_impl(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}
#else
inline __device__ int dp4a_impl(int a, int b, int c) {
    const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + a_bytes[0] * b_bytes[0] + a_bytes[1] * b_bytes[1] +
               a_bytes[2] * b_bytes[2] + a_bytes[3] * b_bytes[3];
}
#endif

// ============================================================================
// Kernel 1: Warp-level for small batch (M < 16)
// Each warp computes one output element, lanes parallelize over K blocks
// ============================================================================
__global__ void __launch_bounds__(256)
w4a32c8_q4_1_warp_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    
    const int num_blocks_k = K / QK4_1;
    const int blocks_per_warp = (num_blocks_k + WARP_SIZE - 1) / WARP_SIZE;
    
    for (int idx = warp_id; idx < M * N; idx += num_warps) {
        const int row = idx / N;
        const int col = idx % N;
        
        float sum = 0.0f;
        
        for (int b_offset = 0; b_offset < blocks_per_warp; ++b_offset) {
            const int kb = b_offset * WARP_SIZE + lane_id;
            if (kb >= num_blocks_k) continue;
            
            const int k_start = kb * QK4_1;
            
            // Load weight block
            const block_q4_1* w_block = &weight[col * num_blocks_k + kb];
            float d_w = half_to_float(w_block->d);
            float m_w = half_to_float(w_block->m);
            
            // Load activation (vectorized float4 loads)
            const float* act_ptr = &activation[row * K + k_start];
            
            float a[32];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4 v = *reinterpret_cast<const float4*>(&act_ptr[i * 4]);
                a[i * 4]     = v.x;
                a[i * 4 + 1] = v.y;
                a[i * 4 + 2] = v.z;
                a[i * 4 + 3] = v.w;
            }
            
            // Q8_1 quantization: find max and sum
            float a_max = fabsf(a[0]);
            float a_sum = a[0];
            #pragma unroll
            for (int i = 1; i < 32; ++i) {
                a_max = fmaxf(a_max, fabsf(a[i]));
                a_sum += a[i];
            }
            
            float d_a = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
            float s_a = a_sum;
            
            // Quantize and pack for DP4A
            int a_packed[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int8_t q0 = (int8_t)__float2int_rn(a[i * 4] / d_a);
                int8_t q1 = (int8_t)__float2int_rn(a[i * 4 + 1] / d_a);
                int8_t q2 = (int8_t)__float2int_rn(a[i * 4 + 2] / d_a);
                int8_t q3 = (int8_t)__float2int_rn(a[i * 4 + 3] / d_a);
                a_packed[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | 
                              ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
            }
            
            // INT8 dot product using DP4A
            int32_t sumi = 0;
            
            // Process low nibbles (first 16 weight values)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t wp = *reinterpret_cast<const uint32_t*>(&w_block->qs[i * 4]);
                int8_t w0 = (int8_t)(wp & 0x0F);
                int8_t w1 = (int8_t)((wp >> 8) & 0x0F);
                int8_t w2 = (int8_t)((wp >> 16) & 0x0F);
                int8_t w3 = (int8_t)((wp >> 24) & 0x0F);
                int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | 
                             ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
                sumi = dp4a_impl(a_packed[i], w_pack, sumi);
            }
            
            // Process high nibbles (next 16 weight values)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t wp = *reinterpret_cast<const uint32_t*>(&w_block->qs[i * 4]);
                int8_t w0 = (int8_t)((wp >> 4) & 0x0F);
                int8_t w1 = (int8_t)((wp >> 12) & 0x0F);
                int8_t w2 = (int8_t)((wp >> 20) & 0x0F);
                int8_t w3 = (int8_t)((wp >> 28) & 0x0F);
                int w_pack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | 
                             ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
                sumi = dp4a_impl(a_packed[i + 4], w_pack, sumi);
            }
            
            // Q4_1 formula: result = d_w * d_a * sumi + m_w * s_a
            sum += d_w * d_a * (float)sumi + m_w * s_a;
        }
        
        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) {
            output[row * N + col] = sum;
        }
    }
}

// ============================================================================
// Kernel 2: Tiled for large batch (M >= 16)
// ============================================================================
__global__ void w4a32c8_q4_1_tiled_kernel(
    const block_q4_1* __restrict__ weight,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int M, const int N, const int K) {
    
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int tid = threadIdx.y * THREADS_N + threadIdx.x;
    const int thread_m = threadIdx.y;
    const int thread_n = threadIdx.x;
    
    __shared__ float smem_act[TILE_M][TILE_K];
    __shared__ int8_t smem_a_q[TILE_M][TILE_K];
    __shared__ float smem_d_a[TILE_M];
    __shared__ float smem_s_a[TILE_M];
    __shared__ block_q4_1 smem_w[TILE_N];
    
    const int items_m = TILE_M / THREADS_M;
    const int items_n = TILE_N / THREADS_N;
    float accum[items_m][items_n];
    
    #pragma unroll
    for (int i = 0; i < items_m; ++i)
        #pragma unroll
        for (int j = 0; j < items_n; ++j)
            accum[i][j] = 0.0f;
    
    const int num_kb = K / QK4_1;
    
    for (int kb = 0; kb < num_kb; ++kb) {
        const int k_off = kb * QK4_1;
        
        // Load activation tile
        const int loads = (TILE_M * TILE_K) / (THREADS_M * THREADS_N);
        #pragma unroll 4
        for (int l = 0; l < loads; ++l) {
            int idx = tid + l * (THREADS_M * THREADS_N);
            int ml = idx / TILE_K, kl = idx % TILE_K;
            int mg = block_m * TILE_M + ml, kg = k_off + kl;
            if (mg < M && kg < K) smem_act[ml][kl] = activation[mg * K + kg];
            else if (ml < TILE_M && kl < TILE_K) smem_act[ml][kl] = 0.0f;
        }
        __syncthreads();
        
        // Quantize activation
        if (thread_m < items_m) {
            for (int mo = 0; mo < TILE_M / items_m; ++mo) {
                int ml = thread_m + mo * items_m;
                if (ml >= TILE_M) continue;
                float mx = 0.0f, sm = 0.0f;
                for (int k = thread_n; k < TILE_K; k += THREADS_N) {
                    mx = fmaxf(mx, fabsf(smem_act[ml][k]));
                    sm += smem_act[ml][k];
                }
                #pragma unroll
                for (int o = THREADS_N / 2; o > 0; o >>= 1) {
                    mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
                    sm += __shfl_down_sync(0xffffffff, sm, o);
                }
                if (thread_n == 0) {
                    smem_d_a[ml] = (mx > 0.0f) ? (mx / 127.0f) : 1.0f;
                    smem_s_a[ml] = sm;
                }
            }
        }
        __syncthreads();
        
        #pragma unroll 4
        for (int l = 0; l < loads; ++l) {
            int idx = tid + l * (THREADS_M * THREADS_N);
            int ml = idx / TILE_K, kl = idx % TILE_K;
            if (ml < TILE_M && kl < TILE_K)
                smem_a_q[ml][kl] = (int8_t)__float2int_rn(smem_act[ml][kl] / smem_d_a[ml]);
        }
        
        for (int nl = tid; nl < TILE_N; nl += THREADS_M * THREADS_N) {
            int ng = block_n * TILE_N + nl;
            if (ng < N) smem_w[nl] = weight[ng * num_kb + kb];
        }
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int i = 0; i < items_m; ++i) {
            int ml = thread_m * items_m + i;
            int mg = block_m * TILE_M + ml;
            if (mg >= M) continue;
            float d_a = smem_d_a[ml], s_a = smem_s_a[ml];
            
            #pragma unroll
            for (int j = 0; j < items_n; ++j) {
                int nl = thread_n * items_n + j;
                int ng = block_n * TILE_N + nl;
                if (ng >= N) continue;
                
                const block_q4_1* wb = &smem_w[nl];
                float d_w = half_to_float(wb->d);
                float m_w = half_to_float(wb->m);
                
                int32_t si = 0;
                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    int ap = *reinterpret_cast<const int*>(&smem_a_q[ml][ii * 4]);
                    uint32_t wp = *reinterpret_cast<const uint32_t*>(&wb->qs[ii * 4]);
                    int8_t w0 = (int8_t)(wp & 0x0F);
                    int8_t w1 = (int8_t)((wp >> 8) & 0x0F);
                    int8_t w2 = (int8_t)((wp >> 16) & 0x0F);
                    int8_t w3 = (int8_t)((wp >> 24) & 0x0F);
                    int wpk = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
                    si = dp4a_impl(ap, wpk, si);
                }
                #pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    int ap = *reinterpret_cast<const int*>(&smem_a_q[ml][16 + ii * 4]);
                    uint32_t wp = *reinterpret_cast<const uint32_t*>(&wb->qs[ii * 4]);
                    int8_t w0 = (int8_t)((wp >> 4) & 0x0F);
                    int8_t w1 = (int8_t)((wp >> 12) & 0x0F);
                    int8_t w2 = (int8_t)((wp >> 20) & 0x0F);
                    int8_t w3 = (int8_t)((wp >> 28) & 0x0F);
                    int wpk = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
                    si = dp4a_impl(ap, wpk, si);
                }
                accum[i][j] += d_w * d_a * (float)si + m_w * s_a;
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < items_m; ++i) {
        int mg = block_m * TILE_M + thread_m * items_m + i;
        if (mg >= M) continue;
        #pragma unroll
        for (int j = 0; j < items_n; ++j) {
            int ng = block_n * TILE_N + thread_n * items_n + j;
            if (ng < N) output[mg * N + ng] = accum[i][j];
        }
    }
}

// ============================================================================
// Host function
// ============================================================================
torch::Tensor forward(torch::Tensor weight, torch::Tensor activation, int M, int N, int K) {
    AT_ASSERTM(activation.dim() == 2, "Activation must be 2D");
    AT_ASSERTM(activation.size(0) == M, "M dimension mismatch");
    AT_ASSERTM(activation.size(1) == K, "K dimension mismatch");
    
    int num_blocks = K / 32;
    int bytes_per_block = 20;
    
    torch::Tensor weight_flat;
    if (weight.dim() == 1) {
        int64_t expected = N * num_blocks * bytes_per_block;
        AT_ASSERTM(weight.size(0) == expected, "Weight size mismatch");
        weight_flat = weight;
    } else if (weight.dim() == 3) {
        AT_ASSERTM(weight.size(0) == N && weight.size(1) == num_blocks && 
                   weight.size(2) == bytes_per_block, "Weight shape mismatch");
        weight_flat = weight.contiguous().view({-1});
    } else {
        AT_ASSERTM(false, "Weight must be 1D or 3D");
    }
    
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));
    
    if (M < BATCH_THRESHOLD) {
        int total = M * N;
        int tpb = 256;
        int warps = tpb / WARP_SIZE;
        int blocks = (total + warps - 1) / warps;
        
        w4a32c8_q4_1_warp_kernel<<<blocks, tpb>>>(
            reinterpret_cast<const block_q4_1*>(weight_flat.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        dim3 threads(THREADS_N, THREADS_M);
        dim3 blocks((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
        
        w4a32c8_q4_1_tiled_kernel<<<blocks, threads>>>(
            reinterpret_cast<const block_q4_1*>(weight_flat.data_ptr<uint8_t>()),
            activation.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
    
    cudaError_t err = cudaGetLastError();
    AT_ASSERTM(err == cudaSuccess, "CUDA error: " + std::string(cudaGetErrorString(err)));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "W4A32C8 Q4_1 GEMM - Final Optimized");
}
