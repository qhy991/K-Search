#!/usr/bin/env python3
"""
GEMM 参考实现模块

提供各种量化格式的矩阵乘法参考实现，用于验证 CUDA kernel 的正确性。
支持的格式：Q8_0 × FP32, Q4_0 × FP32, Q4_0 × Q8_1, Q4_1 × FP32, Q4_1 × Q8_1
"""

import numpy as np
import struct


def reference_q8_0_gemm(weight_q, activation, M, N, K):
    """FP32 参考实现：反量化 Q8_0 权重后计算 GEMM（作为 ground truth）

    Args:
        weight_q: 量化的 Q8_0 权重 (uint8 tensor)
        activation: FP32 激活值 (float32 tensor)
        M, N, K: 矩阵维度

    Returns:
        FP32 输出张量，形状 [M, N]
    """
    import torch

    num_blocks = K // 32
    weight_np = weight_q.cpu().numpy()
    activation_np = activation.cpu().float().numpy()

    # 反量化权重：Q8_0 → FP32
    weight_fp32 = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for b in range(num_blocks):
            offset = (n * num_blocks + b) * 34
            # 读取 FP16 scale
            scale_bytes = weight_np[offset:offset + 2]
            d_w = struct.unpack('<e', bytes(scale_bytes))[0]
            # 读取 INT8 量化值
            qs_bytes = weight_np[offset + 2:offset + 34]
            qs = np.frombuffer(qs_bytes, dtype=np.int8)
            # 反量化
            k_start = b * 32
            weight_fp32[n, k_start:k_start + 32] = qs.astype(np.float32) * d_w

    # 直接 FP32 矩阵乘法作为 ground truth
    output = activation_np @ weight_fp32.T  # [M, K] @ [K, N] = [M, N]

    return torch.from_numpy(output.astype(np.float32)).to(activation.device)


def reference_q4_0_fp32_gemm(weight_q, activation_fp32, M, N, K):
    """Q4_0 × FP32 GEMM 参考实现（反量化后做 FP32 矩阵乘法）

    Q4_0 编码：q = round(val / scale + 8)，q ∈ [0, 15]
    Q4_0 解码：val = scale × (q - 8)，val ∈ [-8×scale, 7×scale]

    计算公式: output = activation @ (d_w × (q_w - 8))^T

    解包方式：llama.cpp 兼容 (先所有低 nibbles，后所有高 nibbles)
    - 位置 0-15: 所有字节的低 nibbles
    - 位置 16-31: 所有字节的高 nibbles

    Args:
        weight_q: Q4_0 量化权重
        activation_fp32: FP32 激活值
        M, N, K: 矩阵维度

    Returns:
        FP32 输出张量，形状 [M, N]
    """
    import torch

    num_blocks = K // 32
    weight_np = weight_q.cpu().numpy()
    activation_np = activation_fp32.cpu().numpy()

    # 解析 Q4_0 weight blocks: [N, num_blocks, 18]
    w_blocks = weight_np.reshape(N, num_blocks, 18)
    # 提取 scale (FP16 → FP32): [N, num_blocks]
    w_scales_raw = w_blocks[:, :, :2].copy()
    w_scales = np.frombuffer(w_scales_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 packed 4-bit 值: [N, num_blocks, 16]
    w_packed = w_blocks[:, :, 2:]
    # 解包为 32 个值: llama.cpp 方式 (先所有低 nibbles，后所有高 nibbles)
    w_low = (w_packed & 0x0F).astype(np.int32)       # [N, num_blocks, 16]
    w_high = ((w_packed >> 4) & 0x0F).astype(np.int32)  # [N, num_blocks, 16]
    # llama.cpp 兼容: 先所有低 nibbles (位置 0-15)，后所有高 nibbles (位置 16-31)
    w_qs = np.concatenate([w_low, w_high], axis=-1)  # [N, num_blocks, 32]

    # 反量化权重：w_fp32 = d_w × (q_w - 8)
    # Q4_0 编码：q = round(val / d_w + 8)，解码：val = d_w × (q - 8)
    W_dequant = (w_scales[:, :, np.newaxis] * (w_qs.astype(np.float32) - 8.0)).reshape(N, K)  # [N, K]

    # FP32 矩阵乘法: C = A @ W^T
    output = activation_np @ W_dequant.T  # [M, K] @ [K, N] -> [M, N]

    return torch.from_numpy(output.astype(np.float32))


def reference_q4_0_q8_1_gemm(weight_q, activation_q, M, N, K):
    """内置的 Q4_0 × Q8_1 GEMM 参考实现（向量化 NumPy 版本）

    计算公式: output[m, n] = sum_b(d_w[n,b] * (d_a[m,b] * sum_i - 8 * s_a[m,b]))

    Args:
        weight_q: Q4_0 量化权重
        activation_q: Q8_1 量化激活
        M, N, K: 矩阵维度

    Returns:
        FP32 输出张量，形状 [M, N]
    """
    import torch

    num_blocks = K // 32
    weight_np = weight_q.cpu().numpy()
    activation_np = activation_q.cpu().numpy()

    # 解析 Q4_0 weight blocks: [N, num_blocks, 18]
    w_blocks = weight_np.reshape(N, num_blocks, 18)
    # 提取 scale (FP16 → FP32): [N, num_blocks]
    w_scales_raw = w_blocks[:, :, :2].copy()
    w_scales = np.frombuffer(w_scales_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 packed 4-bit 值: [N, num_blocks, 16]
    w_packed = w_blocks[:, :, 2:]
    # 解包为 32 个值: llama.cpp 方式 (先所有低 nibbles，后所有高 nibbles)
    w_low = (w_packed & 0x0F).astype(np.int32)       # [N, num_blocks, 16]
    w_high = ((w_packed >> 4) & 0x0F).astype(np.int32)  # [N, num_blocks, 16]
    # llama.cpp 兼容: 先所有低 nibbles (位置 0-15)，后所有高 nibbles (位置 16-31)
    w_qs = np.concatenate([w_low, w_high], axis=-1)  # [N, num_blocks, 32]

    # 解析 Q8_1 activation blocks: [M, num_blocks, 36]
    a_blocks = activation_np.reshape(M, num_blocks, 36)
    # 提取 d 和 s (half2): [M, num_blocks]
    a_ds_raw = a_blocks[:, :, :4].copy()
    a_ds = np.frombuffer(a_ds_raw.tobytes(), dtype=np.float16).reshape(M, num_blocks, 2).astype(np.float32)
    a_d = a_ds[:, :, 0]  # [M, num_blocks]
    a_s = a_ds[:, :, 1]  # [M, num_blocks]
    # 提取量化值: [M, num_blocks, 32]
    a_qs = a_blocks[:, :, 4:].view(np.int8).astype(np.int32)

    # 逐块向量化计算
    output = np.zeros((M, N), dtype=np.float64)
    for b in range(num_blocks):
        # sumi[m, n] = sum_i(w_qs[n, b, i] * a_qs[m, b, i])
        sumi_b = a_qs[:, b, :] @ w_qs[:, b, :].T  # [M, N]
        # d_w * (d_a * sumi - 8.0 * s_a)
        d_w_b = w_scales[:, b]  # [N]
        d_a_b = a_d[:, b:b+1]  # [M, 1]
        s_a_b = a_s[:, b:b+1]  # [M, 1]
        output += d_w_b[np.newaxis, :] * (d_a_b * sumi_b - 8.0 * s_a_b)

    return torch.from_numpy(output.astype(np.float32))


def reference_q4_1_fp32_gemm(weight_q, activation_fp32, M, N, K):
    """内置的 Q4_1 × FP32 GEMM 参考实现（W4A32C8：反量化后做 FP32 矩阵乘法）

    反量化公式: w_fp32 = d * w_vals + m

    Args:
        weight_q: Q4_1 量化权重
        activation_fp32: FP32 激活值
        M, N, K: 矩阵维度

    Returns:
        FP32 输出张量，形状 [M, N]
    """
    import torch

    num_blocks = K // 32
    weight_np = weight_q.cpu().numpy()

    # 解析 Q4_1 weight blocks: [N, num_blocks, 20]
    w_blocks = weight_np.reshape(N, num_blocks, 20)
    # 提取 scale d (FP16 → FP32): [N, num_blocks]
    w_scales_raw = w_blocks[:, :, :2].copy()
    w_scales = np.frombuffer(w_scales_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 min m (FP16 → FP32): [N, num_blocks]
    w_mins_raw = w_blocks[:, :, 2:4].copy()
    w_mins = np.frombuffer(w_mins_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 packed 4-bit 值: [N, num_blocks, 16]
    w_packed = w_blocks[:, :, 4:]
    # 解包为 32 个值: llama.cpp 方式 (先所有低 nibbles，后所有高 nibbles)
    w_low = (w_packed & 0x0F).astype(np.float32)       # [N, num_blocks, 16]
    w_high = ((w_packed >> 4) & 0x0F).astype(np.float32)  # [N, num_blocks, 16]
    # llama.cpp 兼容: 先所有低 nibbles (位置 0-15)，后所有高 nibbles (位置 16-31)
    w_vals = np.concatenate([w_low, w_high], axis=-1)  # [N, num_blocks, 32]

    # 反量化：w_fp32 = d * w_vals + m
    W_dequant = (w_scales[:, :, np.newaxis] * w_vals + w_mins[:, :, np.newaxis]).reshape(N, K)  # [N, K]

    # FP32 矩阵乘法: C = A @ W^T
    act_np = activation_fp32.cpu().numpy().astype(np.float32)
    if act_np.ndim == 1:
        act_np = act_np.reshape(1, -1)
    output = act_np @ W_dequant.T  # [M, K] @ [K, N] -> [M, N]

    return torch.from_numpy(output.astype(np.float32))


def reference_q4_1_dynamic_q8_1_gemm(weight_q, activation_fp32, M, N, K):
    """Q4_1 × FP32 动态量化 GEMM 参考实现

    激活为是 FP32，在函数内部动态量化为 Q8_1，然后使用公式计算

    计算公式: output[m, n] = sum_b(d_w[n,b] * d_a[m,b] * sum_i + m_w[n,b] * s_a[m,b])

    Args:
        weight_q: Q4_1 量化权重
        activation_fp32: FP32 激活值
        M, N, K: 矩阵维度

    Returns:
        FP32 输出张量
形状 [M, N]
    """
    import torch

    num_blocks = K // 32
    weight_np = weight_q.cpu().numpy()
    activation_np = activation_fp32.cpu().numpy()

    # 解析 Q4_1 weight blocks: [N, num_blocks, 20]
    w_blocks = weight_np.reshape(N, num_blocks, 20)
    # 提取 scale (FP16 → FP32): [N, num_blocks]
    w_scales_raw = w_blocks[:, :, :2].copy()
    w_scales = np.frombuffer(w_scales_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 min (FP16 → FP32): [N, num_blocks]
    w_mins_raw = w_blocks[:, :, 2:4].copy()
    w_mins = np.frombuffer(w_mins_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 packed 4-bit 值: [N, num_blocks, 16]
    w_packed = w_blocks[:, :, 4:]
    # 解包为 32 个值: llama.cpp 方式 (先所有低 nibbles，后所有高 nibbles)
    w_low = (w_packed & 0x0F).astype(np.int32)       # [N, num_blocks, 16]
    w_high = ((w_packed >> 4) & 0x0F).astype(np.int32)  # [N, num_blocks, 16]
    # llama.cpp 兼容: 先所有低 nibbles (位置 0-15)，后所有高 nibbles (位置 16-31)
    w_qs = np.concatenate([w_low, w_high], axis=-1)  # [N, num_blocks, 32]

    # 对 FP32 激活进行动态 Q8_1 量化
    # 将激活 reshape 为 [M, num_blocks, 32]
    act_blocks = activation_np.reshape(M, num_blocks, 32)

    # 逐块计算
    output = np.zeros((M, N), dtype=np.float64)

    for b in range(num_blocks):
        # 计算当前块的激活统计量
        act_block = act_blocks[:, b, :]  # [M, 32]

        # 动态量化: 计算 scale (d8_1) 和 sum (s8_1)
        act_max = np.max(np.abs(act_block), axis=1)  # [M]
        act_sum = np.sum(act_block, axis=1)  # [M]

        d_a = act_max / 127.0
        d_a = np.maximum(d_a, 1e-6)  # 避免 scale 为 0

        # 将 FP32 激活量化为 INT8
        a_qs = np.clip(np.round(act_block / d_a[:, np.newaxis]), -128,127).astype(np.int32)  # [M, 32]

        # 计算 sumi = sumi_b = a_qs @ w_qs[:, b, :].T  # [M, N]

        # 应用公式: d_w * d_a * sumi + m_w * s_a
        d_w_b = w_scales[:, b]  # [N]
        m_w_b = w_mins[:, b]    # [N]
        d_a_b = d_a[:, np.newaxis]  # [M, 1]
        s_a_b = act_sum[:, np.newaxis]  # [M, 1]

        output += d_w_b[np.newaxis, :] * d_a_b * sumi_b + m_w_b[np.newaxis, :] * s_a_b

    return torch.from_numpy(output.astype(np.float32))


def reference_q4_1_q8_1_gemm(weight_q, activation_q, M, N, K):
    """内置的 Q4_1 × Q8_1 GEMM 参考实现（向量化 NumPy 版本）

    计算公式: output[m, n] = sum_b(d_w[n,b] * d_a[m,b] * sum_i + m_w[n,b] * s_a[m,b])

    Args:
        weight_q: Q4_1 量化权重
        activation_q: Q8_1 量化激活
        M, N, K: 矩阵维度

    Returns:
        FP32 输出张量，形状 [M, N]
    """
    import torch

    num_blocks = K // 32
    weight_np = weight_q.cpu().numpy()
    activation_np = activation_q.cpu().numpy()

    # 解析 Q4_1 weight blocks: [N, num_blocks, 20]
    w_blocks = weight_np.reshape(N, num_blocks, 20)
    # 提取 scale (FP16 → FP32): [N, num_blocks]
    w_scales_raw = w_blocks[:, :, :2].copy()
    w_scales = np.frombuffer(w_scales_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 min (FP16 → FP32): [N, num_blocks]
    w_mins_raw = w_blocks[:, :, 2:4].copy()
    w_mins = np.frombuffer(w_mins_raw.tobytes(), dtype=np.float16).reshape(N, num_blocks).astype(np.float32)
    # 提取 packed 4-bit 值: [N, num_blocks, 16]
    w_packed = w_blocks[:, :, 4:]
    # 解包为 32 个值: llama.cpp 方式 (先所有低 nibbles，后所有高 nibbles)
    w_low = (w_packed & 0x0F).astype(np.int32)       # [N, num_blocks, 16]
    w_high = ((w_packed >> 4) & 0x0F).astype(np.int32)  # [N, num_blocks, 16]
    # llama.cpp 兼容: 先所有低 nibbles (位置 0-15)，后所有高 nibbles (位置 16-31)
    w_qs = np.concatenate([w_low, w_high], axis=-1)  # [N, num_blocks, 32]

    # 解析 Q8_1 activation blocks: [M, num_blocks, 36]
    a_blocks = activation_np.reshape(M, num_blocks, 36)
    # 提取 d 和 s (half2): [M, num_blocks]
    a_ds_raw = a_blocks[:, :, :4].copy()
    a_ds = np.frombuffer(a_ds_raw.tobytes(), dtype=np.float16).reshape(M, num_blocks, 2).astype(np.float32)
    a_d = a_ds[:, :, 0]  # [M, num_blocks]
    a_s = a_ds[:, :, 1]  # [M, num_blocks]
    # 提取量化值: [M, num_blocks, 32]
    a_qs = a_blocks[:, :, 4:].view(np.int8).astype(np.int32)

    # 逐块向量化计算
    output = np.zeros((M, N), dtype=np.float64)
    for b in range(num_blocks):
        # sumi[m, n] = sum_i(w_qs[n, b, i] * a_qs[m, b, i])
        sumi_b = a_qs[:, b, :] @ w_qs[:, b, :].T  # [M, N]
        # d_w * d_a * sumi + m_w * s_a
        d_w_b = w_scales[:, b]  # [N]
        m_w_b = w_mins[:, b]    # [N]
        d_a_b = a_d[:, b:b+1]  # [M, 1]
        s_a_b = a_s[:, b:b+1]  # [M, 1]
        output += d_w_b[np.newaxis, :] * d_a_b * sumi_b + m_w_b[np.newaxis, :] * s_a_b

    return torch.from_numpy(output.astype(np.float32))
