import torch
import struct

@torch.no_grad()
def run(activation, weight):
    """
    Reference implementation for W4A32C8 Q4_0×FP32_INT8 GEMM.

    This implementation matches the llama.cpp Q4_0 × Q8_1 style pattern:
    1. Quantize FP32 activation to Q8_1 style per-block (32 values)
    2. Compute INT8 dot product with Q4_0 weights
    3. Apply scales with offset compensation: result = d4_0 * (d_a * sumi - 8 * s_a)

    The -8*s_a term compensates for Q4_0's offset-8 encoding.

    Args:
        activation: FP32 activation tensor [M, K]
        weight: Q4_0 quantized weight tensor [N, K/32, 18]

    Returns:
        FP32 output tensor [M, N]
    """
    M, K = activation.shape
    N = weight.shape[0]
    num_blocks = K // 32

    # Move to CPU for byte-level access
    weight_cpu = weight.cpu().numpy()
    activation_cpu = activation.cpu().float().numpy()

    output = torch.zeros(M, N, dtype=torch.float32, device='cpu')

    for m in range(M):
        for n in range(N):
            acc = 0.0
            for b in range(num_blocks):
                # === Unpack Q4_0 weight block ===
                block_data = weight_cpu[n, b, :]
                d_w = struct.unpack('<e', bytes(block_data[0:2]))[0]

                # Q4_0: 32 packed 4-bit values stored in 16 bytes
                # Each byte contains two 4-bit values (low nibble = first, high nibble = second)
                w_qs = []
                for i in range(16):
                    byte_val = block_data[2 + i]
                    low = byte_val & 0x0F       # values 0-15
                    high = (byte_val >> 4) & 0x0F  # values 0-15
                    w_qs.extend([low, high])

                # === Quantize activation block to Q8_1 style ===
                k_start = b * 32

                # Find activation scale for this block
                a_max = 0.0
                for i in range(32):
                    a_val = activation_cpu[m, k_start + i]
                    a_max = max(a_max, abs(a_val))
                d_a = a_max / 127.0 if a_max > 0 else 1.0

                # Compute activation sum (s_a) for Q8_1 style
                # s_a is the sum of FP32 values (NOT scaled by d_a)
                a_sum = 0.0
                for i in range(32):
                    a_val = activation_cpu[m, k_start + i]
                    a_sum += a_val

                # Quantize activation values to int8
                a_qs = []
                for i in range(32):
                    a_val = activation_cpu[m, k_start + i]
                    a_int8 = int(round(a_val / d_a))
                    a_int8 = max(-128, min(127, a_int8))
                    a_qs.append(a_int8)

                # === INT8 dot product with Q4_0 unpacking ===
                # Q4_0 stores values 0-15, representing actual values -8 to +7
                # w_qs = [low0, high0, low1, high1, ...] where each low/high is in [0, 15]
                # The dot product pattern is: low[i] * a[i] + high[i] * a[i+16]
                sumi = 0
                for i in range(16):
                    # w_qs[2*i] = low nibble of byte i (0-15)
                    # w_qs[2*i+1] = high nibble of byte i (0-15)
                    w_low = int(w_qs[2*i])
                    w_high = int(w_qs[2*i+1])
                    sumi += w_low * a_qs[i]       # low nibble * a[i]
                    sumi += w_high * a_qs[i+16]   # high nibble * a[i+16]

                # Apply scales (llama.cpp formula: d4_0 * (d_a * sumi - 8 * s_a))
                # The -8*s_a term compensates for the offset in Q4_0 encoding
                acc += d_w * (d_a * sumi - 8.0 * a_sum)

            output[m, n] = float(acc)

    return output.to(activation.device)


def run_fp32_reference(weight, activation, **params):
    """
    Simple FP32 matmul reference for testing.

    Args:
        weight: FP32 weight [N, K]
        activation: FP32 activation [M, K]

    Returns:
        output: [M, N] = activation @ weight.T
    """
    return torch.matmul(activation, weight.T)
