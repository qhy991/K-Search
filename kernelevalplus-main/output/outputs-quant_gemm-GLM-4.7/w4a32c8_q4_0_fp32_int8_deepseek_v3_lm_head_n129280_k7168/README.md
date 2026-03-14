# W4A32C8 Q4_0 量化 GEMM - DeepSeek-V3 LM Head

**最佳性能**: 2.929 TFLOPS (M=8)

## 快速开始

### 编译

```bash
nvcc -O3 -use_fast_math -gencode=arch=compute_89,code=sm_89 \
    -std=c++17 -shared -Xcompiler -fPIC \
    kernel_best.cu -o w4a32c8_q4_0_deepseek_v3_lm_head.so
```

### 测试

```bash
python llm_kernel_test/unified_test_runner.py --test \
    --definition definitions/quant_gemm/deepseek_v3/w4a32c8_q4_0_fp32_int8_deepseek_v3_lm_head_n129280_k7168.json \
    --attempt-path attempts/w4a32c8_q4_0_fp32_int8_deepseek_v3_lm_head_n129280_k7168_v8
```

## 性能

| 批次 M | 延迟 (ms) | TFLOPS | 相对 V1 |
|--------|-----------|--------|---------|
| 1      | 0.645     | 2.871  | 3.07x   |
| 8      | 5.061     | 2.929  | 3.04x   |
| 512    | 324.804   | 2.922  | 3.02x   |

## 文件

- `kernel_best.cu` - 最佳实现 (V8)
- `test_results.json` - 测试结果
- `summary.md` - 优化总结 (中文)
- `VERSIONS.md` - 版本迭代记录
- `README.md` - 本文件

## 关键优化

1. **共享内存激活量化缓存** - 避免重复计算
2. **DP4A INT8 点积** - 硬件加速
3. **Warp 级并行** - 32 线程协作
4. **Warp 级归约** - 高效求和
5. **统一架构** - 单一 kernel 处理所有批次

## 正确性

所有测试配置均通过 (NMSE < 0.05):
- single_token: NMSE = 0.000355 ✓
- small_batch: NMSE = 0.000239 ✓
- large_batch: NMSE = 0.000240 ✓

## 硬件

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- SM Count: 128

## 问题定义

- N = 129280 (输出特征)
- K = 7168 (输入特征)
- M = 1-512 (批次大小)

量化格式: Q4_0 (每 32 值 18 字节)
