# W8A8C8 Q8_0×Q8_1 扩展测试用例

## 概述

基于成功的 DeepSeek-V3 W8A8C8 测试，我们扩展了测试用例集，覆盖更多实际场景。

## 原始测试用例（5个）

| 名称 | M | N | K | 场景 | 状态 |
|------|---|---|---|------|------|
| single_token | 1 | 7168 | 7168 | 单 token 生成 | ✅ PASS |
| small_batch | 16 | 7168 | 7168 | 小批量推理 | ✅ PASS |
| att_qkv | 1 | 21504 | 7168 | Attention QKV | ✅ PASS |
| moe_up | 8 | 18432 | 7168 | MoE Up 投影 | ✅ PASS |
| moe_down | 8 | 7168 | 18432 | MoE Down 投影 | ✅ PASS |

## 新增测试用例（18个）

### 1. 批量大小变化（5个）

测试不同批量大小对性能的影响：

| 名称 | M | N | K | 场景 |
|------|---|---|---|------|
| medium_batch | 32 | 7168 | 7168 | 中等批量 |
| large_batch | 64 | 7168 | 7168 | 大批量 |
| prefill_128 | 128 | 7168 | 7168 | Prefill 128 tokens |
| prefill_256 | 256 | 7168 | 7168 | Prefill 256 tokens |
| prefill_512 | 512 | 7168 | 7168 | Prefill 512 tokens |

**目的：**
- 验证 M 维度扩展的正确性
- 分析 memory-bound → compute-bound 转换点
- 测试 prefill 阶段性能

### 2. Attention 层变体（4个）

测试不同批量的 Attention 层：

| 名称 | M | N | K | 场景 |
|------|---|---|---|------|
| att_qkv_batch8 | 8 | 21504 | 7168 | QKV batch 8 |
| att_qkv_batch16 | 16 | 21504 | 7168 | QKV batch 16 |
| att_out_batch8 | 8 | 7168 | 7168 | Output batch 8 |
| att_out_batch32 | 32 | 7168 | 7168 | Output batch 32 |

**目的：**
- 测试 N=21504 的大输出维度
- 验证 Attention 层的批量处理

### 3. MoE 层变体（4个）

测试不同批量的 MoE 层：

| 名称 | M | N | K | 场景 |
|------|---|---|---|------|
| moe_up_batch1 | 1 | 18432 | 7168 | Up 单 token |
| moe_up_batch16 | 16 | 18432 | 7168 | Up batch 16 |
| moe_down_batch1 | 1 | 7168 | 18432 | Down 单 token |
| moe_down_batch16 | 16 | 7168 | 18432 | Down batch 16 |

**目的：**
- 测试 MoE 专家网络的不同批量
- 验证 K=18432 的大输入维度

### 4. 方阵测试（3个）

测试方阵配置：

| 名称 | M | N | K | 场景 |
|------|---|---|---|------|
| square_small | 32 | 32 | 32 | 小方阵 |
| square_medium | 512 | 512 | 512 | 中方阵 |
| square_large | 1024 | 1024 | 1024 | 大方阵 |

**目的：**
- 验证非 DeepSeek-V3 特定尺寸
- 测试通用 GEMM 性能

### 5. 极限测试（3个）

测试极端配置：

| 名称 | M | N | K | 场景 |
|------|---|---|---|------|
| extreme_m1 | 1 | 32768 | 7168 | 极大 N |
| extreme_k | 8 | 7168 | 32768 | 极大 K |
| extreme_batch | 1024 | 7168 | 7168 | 极大 M |

**目的：**
- 压力测试
- 发现边界条件问题
- 测试内存限制

## 预期性能特征

### Memory-bound vs Compute-bound

- **M < 32**: Memory-bound（带宽受限）
- **M >= 32**: Compute-bound（计算受限）

### GFLOPS 预期

基于原始测试结果：

| M 范围 | 预期 GFLOPS | 说明 |
|--------|-------------|------|
| M = 1 | 80-110 | Memory-bound |
| M = 8-16 | 250-280 | 过渡区 |
| M >= 32 | 300-400 | Compute-bound |
| M >= 128 | 400-500 | 峰值性能 |

### 精度预期

所有测试应满足：
- **NMSE < 0.05** (阈值)
- **实际 NMSE ≈ 10^-14** (接近浮点精度极限)

## 运行测试

### 完整测试（23个配置）

```bash
cd /home/haiyan/Agent4Kernel/KernelEvalPlus/python
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --benchmark
```

### 快速验证（只测试正确性）

```bash
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1
```

### 测试特定配置

```bash
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --config "M=128,N=7168,K=7168" \
    --benchmark
```

## 预期输出

```
============================================================
 Testing: w8a8c8_q8_0_q8_1
============================================================
Folder: operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1
Module: quant_gemm._C
Device: cuda

Configs: 23

------------------------------------------------------------
 Correctness Tests
------------------------------------------------------------
[PASS] single_token: nmse=2.7485e-14 (threshold=0.05)
[PASS] small_batch: nmse=2.8284e-14 (threshold=0.05)
[PASS] att_qkv: nmse=2.8147e-14 (threshold=0.05)
[PASS] moe_up: nmse=2.7788e-14 (threshold=0.05)
[PASS] moe_down: nmse=4.4176e-14 (threshold=0.05)
[PASS] medium_batch: nmse=...
[PASS] large_batch: nmse=...
... (18 more tests)

Results: 23 passed, 0 failed

------------------------------------------------------------
 Benchmarks
------------------------------------------------------------
single_token         M=    1 N= 7168 K= 7168 |    1.195 ms |    85.97 GFLOPS
small_batch          M=   16 N= 7168 K= 7168 |    6.039 ms |   272.25 GFLOPS
att_qkv              M=    1 N=21504 K= 7168 |    2.890 ms |   106.66 GFLOPS
moe_up               M=    8 N=18432 K= 7168 |    7.766 ms |   272.21 GFLOPS
moe_down             M=    8 N= 7168 K=18432 |   14.013 ms |   150.86 GFLOPS
medium_batch         M=   32 N= 7168 K= 7168 |   ~10 ms    |   ~330 GFLOPS
large_batch          M=   64 N= 7168 K= 7168 |   ~18 ms    |   ~370 GFLOPS
prefill_128          M=  128 N= 7168 K= 7168 |   ~35 ms    |   ~380 GFLOPS
... (15 more benchmarks)

============================================================
```

## 分析维度

### 1. 正确性分析
- 所有配置的 NMSE 应 < 10^-13
- 验证量化误差在可接受范围

### 2. 性能分析
- M 维度扩展的性能曲线
- N/K 维度变化的影响
- Memory-bound vs Compute-bound 转换点

### 3. 可扩展性分析
- 批量大小对吞吐量的影响
- 极限配置的稳定性
- 内存使用情况

## 故障排查

### 如果测试失败

```bash
# 启用详细输出
export CUDA_LAUNCH_BLOCKING=1
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --verbose
```

### 如果测试太慢

```bash
# 只测试小配置
python test_operator.py w8a8c8_q8_0_q8_1 \
    operators/quant_gemm/variants/deepseek_v3/w8a8c8_q8_0_q8_1 \
    --config "M=1,N=32,K=32" \
    --config "M=8,N=64,K=64"
```

## 总结

扩展后的测试集：
- ✅ **23 个测试配置**（原 5 个 + 新 18 个）
- ✅ **覆盖 5 个场景类别**
- ✅ **从单 token 到 1024 batch**
- ✅ **从 32×32 到 32768 维度**

这个测试集能够：
1. 全面验证 kernel 正确性
2. 分析性能特征
3. 发现边界条件问题
4. 为性能优化提供数据

---

**创建时间：** 2026-02-03
**状态：** ✅ 配置已添加到 spec.json
