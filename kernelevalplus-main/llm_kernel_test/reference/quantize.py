#!/usr/bin/env python3
"""
量化函数模块

提供将 FP32 张量转换为各种量化格式的函数。
支持的量化格式：Q8_0, Q4_0, Q4_1, Q8_1
"""

import numpy as np


def quantize_to_q8_0(weight_fp32):
    """将 FP32 权重量化为 Q8_0 格式（向量化版本）

    Q8_0 格式：每个 block 32 个值，使用 FP16 scale

    Args:
        weight_fp32: 形状为 [N, K] 的 FP32 numpy 数组或 torch Tensor

    Returns:
        形状为 [N * num_blocks * 34] 的 uint8 数组（扁平化）
    """
    import torch
    if torch.is_tensor(weight_fp32):
        weight_fp32 = weight_fp32.cpu().numpy()

    N, K = weight_fp32.shape
    num_blocks = K // 32

    weight_np = weight_fp32.reshape(N, num_blocks, 32)

    # 计算每个 block 的 scale: [N, num_blocks]
    max_vals = np.max(np.abs(weight_np), axis=2)
    scales = np.where(max_vals > 0, max_vals / 127.0, 1.0)

    # 量化: [N, num_blocks, 32]
    quantized = np.round(weight_np / scales[:, :, np.newaxis]).clip(-128, 127).astype(np.int8)

    # 构建 Q8_0 blocks: [N, num_blocks, 34]
    weight_q8_0 = np.zeros((N, num_blocks, 34), dtype=np.uint8)

    # Scale → FP16 → 2 bytes (逐个填充以避免 view 问题)
    for n in range(N):
        for b in range(num_blocks):
            scale_fp16 = scales[n, b].astype(np.float16)
            scale_bytes = scale_fp16.tobytes()
            weight_q8_0[n, b, 0] = scale_bytes[0]
            weight_q8_0[n, b, 1] = scale_bytes[1]

    # Quantized values → 32 bytes
    weight_q8_0[:, :, 2:] = quantized.view(np.uint8)

    return weight_q8_0.reshape(-1)


def quantize_to_q4_0(weight_fp32):
    """将 FP32 权重量化为 Q4_0 格式 (18 bytes/block)（向量化版本）

    Q4_0 格式：每个 block 32 个 4-bit 值，使用 FP16 scale
    量化范围: [0, 15]，量化公式: q = round(val / scale + 8)

    Args:
        weight_fp32: 形状为 [N, K] 的 FP32 numpy 数组或 torch Tensor

    Returns:
        形状为 [N * num_blocks * 18] 的 uint8 数组（扁平化）
    """
    import torch
    if torch.is_tensor(weight_fp32):
        weight_fp32 = weight_fp32.cpu().numpy()

    N, K = weight_fp32.shape
    num_blocks = K // 32

    weight_np = weight_fp32.reshape(N, num_blocks, 32)

    # Q4_0 scale: max_abs / 7.0: [N, num_blocks]
    max_vals = np.max(np.abs(weight_np), axis=2)
    scales = np.where(max_vals > 0, max_vals / 7.0, 1.0)

    # 量化到 [0, 15]：q = round(val / scale + 8), clamped to [0, 15]
    quantized = np.round(weight_np / scales[:, :, np.newaxis] + 8.0).clip(0, 15).astype(np.uint8)

    # 构建 Q4_0 blocks: [N, num_blocks, 18]
    weight_q4_0 = np.zeros((N, num_blocks, 18), dtype=np.uint8)

    # Scale → FP16 → 2 bytes (逐个填充以避免 view 问题)
    for n in range(N):
        for b in range(num_blocks):
            scale_fp16 = scales[n, b].astype(np.float16)
            scale_bytes = scale_fp16.tobytes()
            weight_q4_0[n, b, 0] = scale_bytes[0]
            weight_q4_0[n, b, 1] = scale_bytes[1]

    # 打包 4-bit 值：byte[i] = quantized[i] (low) | quantized[i+16] (high) << 4
    # quantized 形状是 [N, num_blocks, 32]
    # 需要打包成 16 个字节：byte[i] = q[i] | (q[i+16] << 4)
    low = quantized[:, :, :16] & 0x0F           # [N, num_blocks, 16] - 位置 0-15
    high = (quantized[:, :, 16:] & 0x0F) << 4    # [N, num_blocks, 16] - 位置 16-31
    weight_q4_0[:, :, 2:] = low | high           # byte[i] = low[i] | high[i]

    return weight_q4_0.reshape(-1)


def quantize_to_q4_1(weight_fp32):
    """将 FP32 权重量化为 Q4_1 格式 (20 bytes/block)（向量化版本）

    Q4_1 格式：每个 block 32 个 4-bit 值，使用 FP16 scale 和 min
    量化范围: [0, 15]，量化公式: q = round((val - min) / scale)

    Args:
        weight_fp32: 形状为 [N, K] 的 FP32 numpy 数组或 torch Tensor

    Returns:
        形状为 [N * num_blocks * 20] 的 uint8 数组（扁平化）
    """
    import torch
    if torch.is_tensor(weight_fp32):
        weight_fp32 = weight_fp32.cpu().numpy()

    N, K = weight_fp32.shape
    num_blocks = K // 32

    weight_np = weight_fp32.reshape(N, num_blocks, 32)

    # Q4_1: 找到每个 block 的 min 和 max
    min_vals = np.min(weight_np, axis=2)
    max_vals = np.max(weight_np, axis=2)

    # Q4_1 scale: (max - min) / 15.0
    scales = np.where(max_vals > min_vals, (max_vals - min_vals) / 15.0, 1.0)

    # 量化到 [0, 15]：q = round((val - min) / scale), clamped to [0, 15]
    # 如果 max == min，所有值量化为 0
    diff = weight_np - min_vals[:, :, np.newaxis]
    quantized = np.round(diff / scales[:, :, np.newaxis]).clip(0, 15).astype(np.uint8)

    # 构建 Q4_1 blocks: [N, num_blocks, 20]
    weight_q4_1 = np.zeros((N, num_blocks, 20), dtype=np.uint8)

    # Scale → FP16 → 2 bytes (逐个填充以避免 view 问题)
    for n in range(N):
        for b in range(num_blocks):
            scale_fp16 = scales[n, b].astype(np.float16)
            scale_bytes = scale_fp16.tobytes()
            weight_q4_1[n, b, 0] = scale_bytes[0]
            weight_q4_1[n, b, 1] = scale_bytes[1]

    # Min → FP16 → 2 bytes (逐个填充以避免 view 问题)
    for n in range(N):
        for b in range(num_blocks):
            min_fp16 = min_vals[n, b].astype(np.float16)
            min_bytes = min_fp16.tobytes()
            weight_q4_1[n, b, 2] = min_bytes[0]
            weight_q4_1[n, b, 3] = min_bytes[1]

    # 打包 4-bit 值：byte[i] = low(位置 i) | high(位置 i+16) << 4
    low = quantized[:, :, :16] & 0x0F
    high = (quantized[:, :, 16:] & 0x0F) << 4
    weight_q4_1[:, :, 4:] = low | high

    return weight_q4_1.reshape(-1)


def quantize_to_q8_1(activation_fp32):
    """将 FP32 激活值量化为 Q8_1 格式 (36 bytes/block)（向量化版本）

    Q8_1 格式：每个 block 32 个 INT8 值，使用 FP16 scale (d) 和 sum (s)
    量化公式: q = round(val / d), s = sum(original_values)

    Args:
        activation_fp32: 形状为 [M, K] 的 FP32 numpy 数组或 torch Tensor

    Returns:
        形状为 [M * num_blocks * 36] 的 uint8 数组（扁平化）
    """
    import torch
    if torch.is_tensor(activation_fp32):
        activation_fp32 = activation_fp32.cpu().numpy()

    M, K = activation_fp32.shape
    num_blocks = K // 32

    activation_np = activation_fp32.reshape(M, num_blocks, 32)

    # Q8_1 scale (d): [M, num_blocks]
    max_vals = np.max(np.abs(activation_np), axis=2)
    d_scales = np.where(max_vals > 0, max_vals / 127.0, 1.0)

    # 量化: [M, num_blocks, 32]
    quantized = np.round(activation_np / d_scales[:, :, np.newaxis]).clip(-128, 127).astype(np.int8)

    # Q8_1 sum (s) = sum of original FP32 values per block: [M, num_blocks]
    s_sums = np.sum(activation_np, axis=2)

    # 构建 Q8_1 blocks: [M, num_blocks, 36]
    activation_q8_1 = np.zeros((M, num_blocks, 36), dtype=np.uint8)

    # ds: d and s packed as half2 → 4 bytes (逐个填充以避免 view 问题)
    for m in range(M):
        for b in range(num_blocks):
            d_fp16 = d_scales[m, b].astype(np.float16)
            d_bytes = d_fp16.tobytes()
            activation_q8_1[m, b, 0] = d_bytes[0]
            activation_q8_1[m, b, 1] = d_bytes[1]
            s_fp16 = s_sums[m, b].astype(np.float16)
            s_bytes = s_fp16.tobytes()
            activation_q8_1[m, b, 2] = s_bytes[0]
            activation_q8_1[m, b, 3] = s_bytes[1]

    # Quantized values → 32 bytes
    activation_q8_1[:, :, 4:] = quantized.view(np.uint8)

    return activation_q8_1.reshape(-1)
