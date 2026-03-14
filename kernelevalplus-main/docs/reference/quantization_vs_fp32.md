# 量化算子 vs 浮点 GEMM：完整对比

本文档总结了量化 GEMM 算子相比浮点 GEMM 在实现上的额外考虑和复杂度。

---

## 一、文件结构对比

### 浮点 GEMM（3 个文件）
```
variants/fp32_gemm/
├── kernel.cu              # ~50-80 行
├── bindings.cpp           # ~30-50 行
└── spec.json              # ~50 行
```

### 量化 GEMM（5 个文件）
```
variants/w8a8c8_q8_0_q8_1/
├── kernel.cu              # ~130 行（+60%）
├── bindings.cpp           # ~30-50 行（相同）
├── spec.json              # ~80 行（+60%）
├── reference.py           # ~86 行（新增）⭐
└── impl.json              # ~7 行（新增）⭐
```

**代码量对比**：
- **简单量化**（W4A16）：106 行（kernel 80 + reference 26）
- **复杂量化**（W8A8C8）：216 行（kernel 130 + reference 86）
- **增长比例**：2-3 倍代码量

---

## 二、新增文件详解

### 1. `reference.py`（自定义参考实现）

**为什么需要**：
- 浮点 GEMM 可以用标准的 `torch.matmul()` 作为参考
- 量化 GEMM 需要**模拟量化过程**，标准 matmul 无法直接使用

**核心功能**：
```python
def run(activation, weight):
    """
    1. 解析量化格式（Q8_0: 每 32 个元素一个块）
    2. 反量化 weight（scale + int8 → float32）
    3. 量化 activation（float32 → scale + int8）
    4. INT8 点积计算
    5. 应用 scale 得到 FP32 结果
    """
    for block in range(K // 32):
        # 解包 Q8_0 weight
        scale_w = unpack_fp16(weight[n, block, 0:2])
        qs_w = unpack_int8(weight[n, block, 2:34])

        # 量化 activation
        scale_a = max(abs(activation[m, block*32:(block+1)*32])) / 127
        qs_a = round(activation / scale_a)

        # INT8 点积
        dot = sum(qs_w[i] * qs_a[i] for i in range(32))

        # 应用 scale
        result += scale_w * scale_a * dot
```

**性能特点**：
- 纯 Python 实现（没有 CUDA 加速）
- 大矩阵（M=32, N=8192, K=8192）需要 40+ 分钟
- 这是为什么 `--benchmark` 很慢的主要原因

### 2. `impl.json`（实现配置）

**作用**：
- 分离**定义**（definition.json）和**实现**（impl.json）
- 避免重复，支持多个实现共享同一定义

**内容**：
```json
{
  "kernel": {
    "file": "kernel.cu",
    "entry_point": "gemm_w8a8c8_q8_0_q8_1"
  },
  "reference": "reference.py:run"
}
```

---

## 三、Kernel 实现的额外考虑

### 1. 数据结构定义

#### 浮点 GEMM
```cuda
// 无需特殊结构
float* A;  // [M, K]
float* B;  // [K, N]
```

#### 量化 GEMM
```cuda
// 需要定义块量化结构
struct block_q8_0 {
    half d;          // FP16 scale (2 bytes)
    int8_t qs[32];   // 32 个 int8 值
};  // 总共 34 bytes/block

struct block_q8_1 {
    half d;          // FP16 scale
    half m;          // FP16 min
    int8_t qs[32];   // 32 个 int8 值
};  // 总共 36 bytes/block
```

### 2. 量化/反量化逻辑

#### 浮点 GEMM
```cuda
// 无需量化
```

#### 量化 GEMM
```cuda
// 在线量化 activation（W8A8 模式）
__device__ void quantize_row_q8_1(
    const float* x,      // 输入 FP32
    block_q8_1* y,       // 输出 Q8_1
    int k
) {
    const int nb = k / 32;

    for (int i = 0; i < nb; i++) {
        // 1. 找最大值
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            amax = fmaxf(amax, fabsf(x[i*32 + j]));
        }

        // 2. 计算 scale
        const float d = amax / 127.0f;
        y[i].d = __float2half(d);

        // 3. 量化
        const float id = d ? 1.0f/d : 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = x[i*32 + j] * id;
            y[i].qs[j] = (int8_t)roundf(v);
        }
    }
}
```

### 3. 计算核心

#### 浮点 GEMM
```cuda
// FP32 乘加
float acc = 0.0f;
for (int k = 0; k < K; k++) {
    acc += A[k] * B[k];  // FP32 × FP32
}
C[idx] = acc;
```

#### 量化 GEMM
```cuda
// INT8 乘加 + scale 处理
int32_t acc_int = 0;  // 必须用 int32 避免溢出
float acc_scale = 0.0f;

// 逐块处理
for (int b = 0; b < num_blocks; b++) {
    // 读取 weight 块
    block_q8_0 w_block = weight[b];
    half scale_w = w_block.d;

    // 量化 activation 块
    block_q8_1 a_block;
    quantize_block(&activation[b*32], &a_block);
    half scale_a = a_block.d;

    // INT8 点积
    int32_t dot = 0;
    for (int i = 0; i < 32; i++) {
        dot += (int32_t)w_block.qs[i] * (int32_t)a_block.qs[i];
    }

    // 累加（应用 scale）
    acc_scale += __half2float(scale_w) * __half2float(scale_a) * (float)dot;
}

C[idx] = acc_scale;
```

### 4. 内存访问模式

#### 浮点 GEMM
```cuda
// 连续访问，简单
float a = A[m * K + k];      // 4 bytes
float b = B[k * N + n];      // 4 bytes
```

#### 量化 GEMM
```cuda
// 块状访问，需要对齐
int block_idx = k / 32;
int in_block_idx = k % 32;

// 读取整个块（34 bytes）
block_q8_0* block_ptr = &weight[n * num_blocks + block_idx];
half scale = block_ptr->d;
int8_t q_val = block_ptr->qs[in_block_idx];

// 或使用共享内存优化
__shared__ block_q8_0 smem_blocks[BLOCK_SIZE];
// 合并访问整个块
```

### 5. 精度控制

#### 浮点 GEMM
```cuda
// FP32 精度，误差小
// NMSE 通常 < 1e-6
```

#### 量化 GEMM
```cuda
// INT8 量化误差
// NMSE 通常 1e-9 到 1e-3
// 需要仔细调整阈值

// 误差来源：
// 1. 量化误差（FP32 → INT8）
// 2. INT8 乘法的舍入误差
// 3. Scale 的 FP16 精度损失
```

---

## 四、spec.json 的差异

### 浮点 GEMM
```json
{
  "name": "fp32_gemm",
  "inputs": {
    "A": {"dtype": "float32", "shape": ["M", "K"]},
    "B": {"dtype": "float32", "shape": ["K", "N"]}
  },
  "outputs": {
    "C": {"dtype": "float32", "shape": ["M", "N"]}
  },
  "reference": "torch.matmul"
}
```

### 量化 GEMM
```json
{
  "name": "w8a8c8_q8_0_q8_1",
  "inputs": {
    "weight": {
      "dtype": "block_q8_0",
      "shape": ["N", "K/32", "34"],
      "quantizer": "quantize_q8_0",
      "block_size": 32
    },
    "activation": {
      "dtype": "float32",
      "shape": ["M", "K"],
      "quantize_online": true
    }
  },
  "outputs": {
    "output": {"dtype": "float32", "shape": ["M", "N"]}
  },
  "reference": "reference.py:run",
  "accuracy": {
    "metric": "nmse",
    "threshold": 0.1
  }
}
```

---

## 五、测试复杂度对比

| 测试项 | 浮点 GEMM | 量化 GEMM | 差异 |
|--------|-----------|-----------|------|
| **正确性验证** | torch.matmul | 自定义 reference.py | +86 行代码 |
| **参考实现速度** | 快（GPU 加速） | 慢（纯 Python） | 540x 慢 |
| **精度阈值** | NMSE < 1e-6 | NMSE < 0.1 | 宽松 10^5 倍 |
| **测试配置** | 简单 shape | 多种 shape（块对齐） | K % 32 == 0 |
| **调试难度** | 低 | 高（量化误差分析） | 需要专业知识 |

---

## 六、性能优化考虑

### 浮点 GEMM 优化点（5 个）
1. Tile 分块
2. 共享内存使用
3. 寄存器复用
4. 向量化访问（float4）
5. Warp shuffle

### 量化 GEMM 额外优化点（+5 个）
6. **量化开销最小化**
   - 在线量化的计算开销
   - 量化可以在 shared memory 中完成

7. **INT8 指令利用**
   - 使用 `dp4a`（INT8 点积指令）
   - Tensor Core INT8 支持（sm_75+）

8. **内存带宽优化**
   - 量化数据更小（1 byte vs 4 bytes）
   - 但需要额外读取 scale（2 bytes/block）
   - 块对齐的内存访问模式

9. **Scale 处理优化**
   - FP16 scale 的快速乘法
   - Scale 可以预先加载到寄存器

10. **混合精度策略**
    - INT8 累加 → INT32 中间结果
    - 最后转换为 FP32 输出

---

## 七、完整对比表

| 维度 | 浮点 GEMM | 量化 GEMM (W8A8C8) | 复杂度增加 |
|------|-----------|-------------------|-----------|
| **文件数量** | 3 | 5 | +67% |
| **代码行数** | ~150 | ~300 | +100% |
| **数据结构** | 简单数组 | 块量化结构 | 需要定义 struct |
| **内存布局** | [M,K] 连续 | [M, K/32, 34] 块状 | 需要块对齐 |
| **计算类型** | FP32 | INT8 → FP32 | 需要类型转换 |
| **量化逻辑** | 无 | 在线量化 | +50-100 行 |
| **参考实现** | torch.matmul | 自定义 Python | +86 行 |
| **测试时间** | 秒级 | 分钟级（含参考） | 540x |
| **精度要求** | NMSE < 1e-6 | NMSE < 0.1 | 宽松 10^5 倍 |
| **调试难度** | 低 | 高 | 需要量化专业知识 |
| **性能优化** | 标准 GEMM | + 量化特定优化 | 额外 5 个优化点 |

---

## 八、关键要点总结

### 🆕 新增文件（2 个）
1. **reference.py**：自定义参考实现（~86 行）
2. **impl.json**：实现配置（~7 行）

### 🔧 额外考虑（10 个方面）
1. 块量化数据结构定义
2. 在线量化/反量化逻辑
3. INT8 计算和溢出处理
4. Scale 的 FP16 精度处理
5. 块对齐的内存访问
6. 量化误差控制
7. 自定义参考实现
8. 更宽松的精度阈值
9. INT8 专用指令优化
10. 混合精度累加策略

### 📈 复杂度增长
- **代码量**：2-3 倍
- **开发时间**：3-5 倍
- **调试难度**：5-10 倍
- **测试时间**：540 倍（含参考实现）

### ✅ 收益
- **内存占用**：减少 75%（4 bytes → 1 byte）
- **理论吞吐量**：提升 4x（INT8 vs FP32）
- **实际加速比**：1.5-3x（取决于实现质量）

---

## 九、实际测试数据

### 性能对比（M=32, N=8192, K=8192）

| 指标 | 测试结果 |
|------|---------|
| **延迟** | 43.6 ms |
| **吞吐量** | 98.4 GFLOPS (0.098 TFLOPS) |
| **理论计算量** | 4.29 TFLOPS |
| **效率** | 2.3% |

**结论**：当前实现还有很大优化空间！

### 测试时间对比

| 测试方法 | 耗时 | 说明 |
|---------|------|------|
| 正确性验证（不含 benchmark） | ~40 分钟 | 主要是参考实现慢 |
| 完整测试（含 benchmark） | ~45 分钟 | +5 分钟 benchmark |
| 快速性能测试（benchmark_only.py） | ~5 秒 | 跳过参考实现 |

**加速比**：540x（使用 benchmark_only.py）

---

## 十、最佳实践

### 开发流程
1. **设计阶段**：定义量化格式和数据结构
2. **实现阶段**：先实现参考实现，再实现 CUDA kernel
3. **测试阶段**：使用 `run_tests.py` 验证正确性
4. **优化阶段**：使用 `benchmark_only.py` 快速迭代

### 工具选择
- **开发调试**：`run_tests.py`（不带 --benchmark）
- **性能优化**：`benchmark_only.py`
- **最终验证**：`run_tests.py --benchmark`

### 注意事项
1. 量化格式必须与参考实现完全一致
2. 注意 INT8 乘法的溢出问题（使用 int32 累加）
3. Scale 的 FP16 精度可能影响最终结果
4. 块对齐要求（K % 32 == 0）

---

## 参考资料

- **快速入门**：[docs/guides/quickstart.md](../guides/quickstart.md)
- **测试指南**：[docs/guides/test_operator_guide.md](../guides/test_operator_guide.md)
- **性能测试**：[docs/guides/benchmark_guide.md](../guides/benchmark_guide.md)
- **工具说明**：[tools/README.md](../../tools/README.md)
