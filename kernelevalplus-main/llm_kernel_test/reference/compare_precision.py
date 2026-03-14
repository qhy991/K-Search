#!/usr/bin/env python3
"""
验证 Q4_0 × FP32 与 Q4_0 × Q8_1 的精度差异

比较两种场景：
1. 计算方式差异（相同输入，不同计算顺序）
2. 激活量化差异（不同输入格式）
"""

import numpy as np
import sys
sys.path.insert(0, '/home/qinhaiyan/kernelevalplus/llm_kernel_test/reference')

def simulate_q4_0_fp32(A_fp32, W_fp32, M=32, N=128, K=128):
    """Q4_0 × FP32: 反量化后计算"""
    from quantize import quantize_to_q4_0

    # 量化权重
    W_q = quantize_to_q4_0(W_fp32.T)  # [K, N] -> 量化后展平

    num_blocks = K // 32
    w_blocks = W_q.reshape(N, num_blocks, 18)

    # 提取 scale (修复 view 操作)
    w_scales = np.zeros((N, num_blocks), dtype=np.float32)
    for n in range(N):
        for b in range(num_blocks):
            offset = (n * num_blocks + b) * 18
            scale_bytes = W_q[offset:offset + 2]
            w_scales[n, b] = np.frombuffer(scale_bytes, dtype=np.float16)[0]

    w_packed = w_blocks[:, :, 2:]

    # llama.cpp 兼容解包
    w_low = (w_packed & 0x0F).astype(np.int32)
    w_high = ((w_packed >> 4) & 0x0F).astype(np.int32)
    w_qs = np.concatenate([w_low, w_high], axis=-1)

    # 反量化
    W_dequant = (w_scales[:, :, np.newaxis] * (w_qs.astype(np.float32) - 8.0)).reshape(N, K)

    # FP32 矩阵乘法
    output = A_fp32 @ W_dequant.T
    return output


def simulate_q4_0_q8_1(A_fp32, W_fp32, M=32, N=128, K=128):
    """Q4_0 × Q8_1: 量化公式计算"""
    from quantize import quantize_to_q4_0, quantize_to_q8_1

    # 量化权重
    W_q = quantize_to_q4_0(W_fp32.T)
    # 量化激活
    A_q = quantize_to_q8_1(A_fp32)

    num_blocks = K // 32

    # 解析 Q4_0 权重
    w_scales = np.zeros((N, num_blocks), dtype=np.float32)
    w_packed = np.zeros((N, num_blocks, 16), dtype=np.uint8)
    for n in range(N):
        for b in range(num_blocks):
            offset = (n * num_blocks + b) * 18
            scale_bytes = W_q[offset:offset + 2]
            w_scales[n, b] = np.frombuffer(scale_bytes, dtype=np.float16)[0]
            w_packed[n, b, :] = W_q[offset + 2:offset + 18]

    w_low = (w_packed & 0x0F).astype(np.int32)
    w_high = ((w_packed >> 4) & 0x0F).astype(np.int32)
    w_qs = np.concatenate([w_low, w_high], axis=-1)

    # 解析 Q8_1 激活
    a_d = np.zeros((M, num_blocks), dtype=np.float32)
    a_s = np.zeros((M, num_blocks), dtype=np.float32)
    a_qs = np.zeros((M, num_blocks, 32), dtype=np.int32)
    for m in range(M):
        for b in range(num_blocks):
            offset = (m * num_blocks + b) * 36
            ds_bytes = A_q[offset:offset + 4]
            ds = np.frombuffer(ds_bytes, dtype=np.float16)
            a_d[m, b] = ds[0]
            a_s[m, b] = ds[1]
            qs_bytes = A_q[offset + 4:offset + 36]
            a_qs[m, b, :] = np.frombuffer(qs_bytes, dtype=np.int8)

    # 量化公式计算
    output = np.zeros((M, N), dtype=np.float64)
    for b in range(num_blocks):
        sumi_b = a_qs[:, b, :] @ w_qs[:, b, :].T
        d_w_b = w_scales[:, b]
        d_a_b = a_d[:, b:b+1]
        s_a_b = a_s[:, b:b+1]
        output += d_w_b[np.newaxis, :] * (d_a_b * sumi_b - 8.0 * s_a_b)

    return output.astype(np.float32)


def simulate_q4_0_fp32_with_quant_formula(A_fp32, W_fp32, M=32, N=128, K=128):
    """Q4_0 × FP32 使用量化公式（但激活不量化）"""
    from quantize import quantize_to_q4_0

    # 量化权重
    W_q = quantize_to_q4_0(W_fp32.T)

    num_blocks = K // 32

    # 解析 Q4_0 权重
    w_scales = np.zeros((N, num_blocks), dtype=np.float32)
    w_packed = np.zeros((N, num_blocks, 16), dtype=np.uint8)
    for n in range(N):
        for b in range(num_blocks):
            offset = (n * num_blocks + b) * 18
            scale_bytes = W_q[offset:offset + 2]
            w_scales[n, b] = np.frombuffer(scale_bytes, dtype=np.float16)[0]
            w_packed[n, b, :] = W_q[offset + 2:offset + 18]

    w_low = (w_packed & 0x0F).astype(np.int32)
    w_high = ((w_packed >> 4) & 0x0F).astype(np.int32)
    w_qs = np.concatenate([w_low, w_high], axis=-1)

    # 使用量化公式（但 d_a=1, s_a=sum(A)）
    output = np.zeros((M, N), dtype=np.float64)
    for b in range(num_blocks):
        sumi_b = A_fp32[:, b*32:(b+1)*32] @ w_qs[:, b, :].T
        d_w_b = w_scales[:, b]
        s_a_b = np.sum(A_fp32[:, b*32:(b+1)*32], axis=1)[:, np.newaxis]
        output += d_w_b[np.newaxis, :] * sumi_b - d_w_b[np.newaxis, :] * 8.0 * s_a_b

    return output.astype(np.float32)


def compute_nmse(output1, output2):
    """计算归一化均方误差"""
    mse = np.mean((output1 - output2) ** 2)
    ref_power = np.mean(output1 ** 2)
    return mse / ref_power if ref_power > 1e-10 else float('inf')


def main():
    print("=" * 70)
    print("Q4_0 × FP32 与 Q4_0 × Q8_1 精度对比实验")
    print("=" * 70)

    np.random.seed(42)

    # 测试配置
    M, N, K = 32, 128, 128

    # 生成测试数据
    W_fp32 = np.random.randn(N, K).astype(np.float32) * 0.1
    A_fp32 = np.random.randn(M, K).astype(np.float32) * 0.1

    # 1. 计算方式对比（相同 FP32 激活）
    print("\n[实验 1] 计算方式差异（都使用 FP32 激活）")
    print("-" * 70)

    output_method1 = simulate_q4_0_fp32(A_fp32, W_fp32, M, N, K)
    output_method2 = simulate_q4_0_fp32_with_quant_formula(A_fp32, W_fp32, M, N, K)

    nmse_method = compute_nmse(output_method1, output_method2)
    max_abs_diff = np.max(np.abs(output_method1 - output_method2))

    print(f"方式1 (反量化后计算) vs 方式2 (量化公式)")
    print(f"  NMSE:        {nmse_method:.2e}")
    print(f"  最大绝对差:  {max_abs_diff:.2e}")
    print(f"  结论: {'✓ 几乎相同' if nmse_method < 1e-6 else '✗ 差异较大'}")

    # 2. 激活量化对比
    print("\n[实验 2] 激活量化差异（Q4_0×FP32 vs Q4_0×Q8_1）")
    print("-" * 70)

    output_fp32_act = simulate_q4_0_fp32(A_fp32, W_fp32, M, N, K)
    output_q8_1_act = simulate_q4_0_q8_1(A_fp32, W_fp32, M, N, K)

    nmse_act_quant = compute_nmse(output_fp32_act, output_q8_1_act)
    max_abs_diff_act = np.max(np.abs(output_fp32_act - output_q8_1_act))

    print(f"Q4_0×FP32 vs Q4_0×Q8_1")
    print(f"  NMSE:        {nmse_act_quant:.2e}")
    print(f"  最大绝对差:  {max_abs_diff_act:.2e}")
    print(f"  相对误差:    {max_abs_diff_act / (np.max(np.abs(output_fp32_act)) + 1e-10) * 100:.2f}%")
    print(f"  结论: 激活量化引入了 {'显著误差' if nmse_act_quant > 1e-4 else '较小误差'}")

    # 3. 统计分析
    print("\n[实验 3] 误差统计分析")
    print("-" * 70)

    abs_error = np.abs(output_fp32_act - output_q8_1_act)
    rel_error = abs_error / (np.abs(output_fp32_act) + 1e-10)

    print(f"绝对误差统计:")
    print(f"  均值:   {np.mean(abs_error):.6f}")
    print(f"  标准差: {np.std(abs_error):.6f}")
    print(f"  中位数: {np.median(abs_error):.6f}")
    print(f"  最大值: {np.max(abs_error):.6f}")

    print(f"\n相对误差统计:")
    print(f"  均值:   {np.mean(rel_error) * 100:.4f}%")
    print(f"  标准差: {np.std(rel_error) * 100:.4f}%")
    print(f"  中位数: {np.median(rel_error) * 100:.4f}%")
    print(f"  最大值: {np.max(rel_error) * 100:.4f}%")

    # 4. 与完全 FP32 的对比（作为基准）
    print("\n[实验 4] 与完全 FP32 GEMM 的对比（基准）")
    print("-" * 70)

    output_fp32_baseline = A_fp32 @ W_fp32.T  # 完全 FP32，无量化

    nmse_q4_0_fp32 = compute_nmse(output_fp32_baseline, output_fp32_act)
    nmse_q4_0_q8_1 = compute_nmse(output_fp32_baseline, output_q8_1_act)

    print(f"完全 FP32 vs Q4_0×FP32:  NMSE = {nmse_q4_0_fp32:.4f}")
    print(f"完全 FP32 vs Q4_0×Q8_1:  NMSE = {nmse_q4_0_q8_1:.4f}")
    print(f"\n结论:")
    print(f"  Q4_0 权量化引入的误差: {nmse_q4_0_fp32 * 100:.2f}%")
    print(f"  Q8_1 激活量化额外引入: {(nmse_q4_0_q8_1 - nmse_q4_0_fp32) * 100:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
