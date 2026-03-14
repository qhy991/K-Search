import torch
import struct

@torch.no_grad()
def run(activation, weight):
    """
    Reference implementation for W8A8C8 Q8_0×Q8_1 GEMM.

    This implementation matches the llama.cpp Q8_0 × Q8_1 pattern:
    1. Quantize FP32 activation to Q8_1 style per-block (32 values)
    2. Compute INT8 dot product with Q8_0 weights
    3. Apply scales: result = d8_0 * d8_1 * sumi

    Note: The Q8_1 sum field is NOT used for Q8_0 weights.
    It's only needed when paired with Q4_0 (which has -8 offset).

    Args:
        activation: FP32 activation tensor [M, K]
        weight: Q8_0 quantized weight tensor [N, K/32, 34]

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
                # === Unpack Q8_0 weight block ===
                block_data = weight_cpu[n, b, :]
                d_w = struct.unpack('<e', bytes(block_data[0:2]))[0]
                w_qs = struct.unpack('<32b', bytes(block_data[2:34]))

                # === Quantize activation block to Q8_1 style ===
                k_start = b * 32

                # Find activation scale for this block
                a_max = 0.0
                for i in range(32):
                    a_val = activation_cpu[m, k_start + i]
                    a_max = max(a_max, abs(a_val))
                d_a = a_max / 127.0 if a_max > 0 else 1.0

                # Quantize activation values to int8
                a_qs = []
                for i in range(32):
                    a_val = activation_cpu[m, k_start + i]
                    a_int8 = int(round(a_val / d_a))
                    a_int8 = max(-128, min(127, a_int8))
                    a_qs.append(a_int8)

                # === INT8 dot product (matching llama.cpp vec_dot_q8_0_q8_1) ===
                # Compute dot product directly on int8 arrays
                sumi = 0
                for i in range(32):
                    sumi += w_qs[i] * a_qs[i]

                # Apply scales (llama.cpp formula: d8_0 * d8_1 * sumi)
                acc += d_w * d_a * sumi

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
