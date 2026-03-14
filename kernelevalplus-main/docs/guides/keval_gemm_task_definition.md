# KernelEval-GEMM 问题定义与 588 个任务的由来

本文档解释 KernelEval-GEMM（量化 GEMM）在本仓库中的**问题定义**，以及为什么一个设备上会有 **588 个 benchmark tasks/cases**。

## 1. 问题定义（Quantized GEMM）

KernelEval-GEMM 评测的是推理场景中最常见的线性层计算：给定 **FP32 激活**与**块量化权重**，计算输出。

### 1.1 张量与维度含义

- `M`：batch 维（可理解为 *batch_size × token_count* 的合并维度；decode 时很小，prefill 时较大）
- `N`：输出维（out_features，对应权重矩阵的行数）
- `K`：输入维（in_features，对应权重矩阵的列数；ggml block quant 约束通常要求 `K % 32 == 0`）

### 1.2 计算公式（以推理线性层为例）

- 激活：`A ∈ R^{M×K}`（存储为 FP32）
- 权重：`W ∈ R^{N×K}`（存储为 ggml block-quant：`Q4_0/Q4_1/Q8_0`）
- 输出：`O ∈ R^{M×N}`（输出 FP32）

计算为：

```
O = A · W^T
```

> 注：在 `llama.cpp/ggml` 的 CUDA 路径中，虽然激活输入常以 FP32 形式提供，但 kernel 内部可能会动态量化为 Q8_1 风格以使用 INT8 dot/DP4A 路径；这不影响本 benchmark 的“输入契约”（输入仍然是 FP32 activation）。

### 1.3 什么是一个 task / case？

一个 task（也可称一个 benchmark case）由四元组定义：

```
(weight_quant, M, N, K)
```

其中 `(N, K)` 来自真实模型的层形状集合，`weight_quant` 为权重量化格式，`M` 为 batch regime。

## 2. 为什么是 588 个 tasks？

KernelEval-GEMM 在一个设备上的 case 数量来自三个维度的笛卡尔积：

1. **28 个真实形状**：`(N, K)` 共 28 组
2. **3 种权重量化**：`Q4_0`、`Q4_1`、`Q8_0`
3. **7 个 batch 取值（M）**：`{1, 2, 3, 4, 5, 8, 512}`

因此：

```
|Tasks| = 28 × 3 × 7 = 588
```

> 注：588 是“每个设备”的 case 数。如果在 `D` 个设备上评测，则总 case 数为 `588 × D`。

## 3. 28 个 (N, K) 形状来自哪里？

这 28 个形状来自主流 LLM/VLA 推理中的典型线性层（attention 投影、FFN 投影、lm_head，以及 MoE 相关投影），强调包含**长尾形状**（thin/tall matrices）。

形状可按模型族理解为：

- Llama3-8B：`(4096,4096) (14336,4096) (4096,14336) (128256,4096)`
- Qwen2.5-7B：`(3584,3584) (18944,3584) (3584,18944) (152064,3584)`
- Qwen3-4B：`(2560,2560) (9728,2560) (2560,9728) (151936,2560)`
- DeepSeek-V2：`(5120,5120)` + MoE/路由相关 `4` 个 + `lm_head`
  - MoE/路由：`(5120,12288) (12288,5120) (1536,5120) (5120,1536)`
  - `lm_head`：`(102400,5120)`
- DeepSeek-V3：`(7168,7168)` + MoE/路由相关 `4` 个 + `lm_head` + 额外长尾 `4` 个
  - MoE/路由：`(7168,18432) (18432,7168) (2048,7168) (7168,2048)`
  - `lm_head`：`(129280,7168)`
  - 额外长尾：`(1536,7168) (7168,1536) (512,7168) (7168,512)`

> 重要澄清：这里的 “MoE” 指的是 **MoE 层内部出现的若干 GEMM（expert up/down、routing 投影等）**。它们仍然是 GEMM task，但来源于 MoE 推理路径。真正的 “MoE op type（端到端）” 还需要额外包含 routing / scatter-gather / combine 等非 GEMM 组成部分。

## 4. 3 种权重量化类型（weight_quant）

KernelEval-GEMM 当前覆盖 ggml/llama.cpp 常用的三种 block-quant 权重格式：

- `Q4_0`：4-bit block quant（offset-8 编码路径常见）
- `Q4_1`：4-bit block quant（带不同 scale/zero-point 布局）
- `Q8_0`：8-bit block quant

激活输入以 `FP32` 提供（对应 ggml baseline 里的 `type_b=f32`）。

## 5. 7 个 batch 取值（M）为什么这样选？

`M` 代表推理运行中的“有效 batch/token 数”：

- `M ∈ {1,2,3,4,5,8}`：近似 decode 阶段的 micro-batch（更关注延迟、launch/occupancy、长尾 shape）
- `M = 512`：代表 prefill/吞吐导向的 batch（更关注带宽/算力上限）

这 7 个点覆盖了从 latency-sensitive 到 throughput-oriented 的两端，并能显著放大不同 `(N,K)` 长尾形状在小 batch 下的性能差异。

## 6. 与 ggml baseline 的维度映射（可选：用于对齐日志/键值）

ggml 的 `test-backend-ops perf` 会输出类似：

```
MUL_MAT(type_a=..., type_b=f32, m, n, k)
```

其中：

- `m = N`
- `n = M`（batch）
- `k = K`

本仓库的基线文件 `core/tools/baseline_data_compact.json` 使用的 key 形如：

```
<prefix>_<type_a>_f32_m{N}_n{M}_k{K}
```

例如（`Q8_0`，`N=7168,K=7168,M=1`）：

```
w8a32c8_q8_0_f32_m7168_n1_k7168
```

## 7. 在仓库里哪里能看到这些定义？

- ggml baseline（包含更多 shape/批次，KernelEval-GEMM 从中选取子集）：`core/tools/baseline_data_compact.json`
- GEMM definitions（包含层/模型标签与 test_configs）：`definitions/quant_gemm/**/*.json`
- Web UI（展示结果与 baseline 对比）：`bench_web/app.py`

