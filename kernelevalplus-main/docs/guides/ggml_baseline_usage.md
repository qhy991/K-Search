# GGML 基线数据使用指南

## 概述

GGML 基线数据系统提供了从 `ggml-python-main` 测试结果中提取的高性能基线数据，用于性能对比和评估。

---

## 快速开始

### 1. 生成基线数据

```bash
# 解析 ggml-python-main 结果
python -m python.tools.ggml_baseline_parser \
    --results-dir /home/haiyan/Agent4Kernel/ggml-python-main/results \
    --output core/tools/baseline_data
```

生成的文件：
- `baseline_data.csv` - CSV 格式
- `baseline_data.json` - JSON 格式（按硬件分组）
- `baseline_data_compact.json` - 紧凑 JSON（按 case_id 索引）

### 2. 查询基线数据

```python
from core.tools.ggml_baseline_api import BaselineAPI

# 初始化 API
api = BaselineAPI()

# 查询基线
baseline = api.get_baseline("RTX4090", "q4_0", 4096, 1, 4096)
print(f"TFLOPS: {baseline['tflops']}")
print(f"GFLOPS: {baseline['gflops']}")
print(f"延迟: {baseline['us_per_run']} us")
```

### 3. 性能比较

```python
# 比较实际性能与基线
result = api.compare_performance(
    hardware="RTX4090",
    type_a="q4_0",
    m=4096, n=1, k=4096,
    actual_tflops=5.2
)

print(f"性能比: {result['ratio']:.2%}")
if result['better']:
    print(f"✅ 比基线快 {result['diff_percent']:.1%}")
else:
    print(f"❌ 比基线慢 {-result['diff_percent']:.1%}")
```

---

## API 参考

### BaselineAPI 类

#### 初始化

```python
api = BaselineAPI(baseline_path="path/to/baseline_data_compact.json")
```

#### 方法

| 方法 | 说明 |
|------|------|
| `get_baseline(hardware, type_a, m, n, k)` | 查询基线数据 |
| `get_by_case_id(case_id, hardware)` | 根据 case_id 查询 |
| `get_closest_baseline(hardware, type_a, m, n, k)` | 获取最接近的配置 |
| `compare_performance(hardware, type_a, m, n, k, actual_tflops)` | 比较性能 |
| `get_all_hardware()` | 获取硬件列表 |
| `print_summary()` | 打印数据摘要 |

---

## 数据格式

### case_id 格式

新格式包含位宽信息：**w{weight}a{activation}c{compute}_{type_a}_{type_b}_m{m}_n{n}_k{k}**

- **w{weight}**: 权重位宽 (4, 5, 8)
- **a{activation}**: 激活位宽 (32=FP32, 16=FP16, 8=INT8)
- **c{compute}**: 计算位宽 (8=INT8 DP4A)
- **type_a**: 量化类型 (q4_0, q4_1, q8_0)
- **type_b**: 激活类型 (f32)
- **m, n, k**: 维度

示例：
- `w4a32c8_q4_0_f32_m4096_n1_k4096` - 4-bit权重, 32-bit激活, 8-bit计算
- `w4a32c8_q4_1_f32_m14336_n1_k4096` - 4-bit非对称权重
- `w8a32c8_q8_0_f32_m7168_n1_k7168` - 8-bit权重

### 紧凑 JSON 格式

```json
{
  "w4a32c8_q4_0_f32_m4096_n1_k4096": {
    "type_a": "q4_0",
    "type_b": "f32",
    "m": 4096,
    "n": 1,
    "k": 4096,
    "hardware": {
      "RTX4090": {
        "tflops": 4.95,
        "gflops": 4950.0,
        "us_per_run": 6.77
      },
      "A100": {
        "tflops": 2.54,
        "gflops": 2540.0,
        "us_per_run": 13.2
      },
      "RTX4070": {
        "tflops": 1.39,
        "gflops": 1390.0,
        "us_per_run": 24.21
      }
    }
  }
}
```

---

## 命令行工具

### 解析结果

```bash
# 解析所有结果
python -m python.tools.ggml_baseline_parser \
    --results-dir /path/to/results

# 只生成 JSON
python -m python.tools.ggml_baseline_parser --json-only

# 指定输出路径
python -m python.tools.ggml_baseline_parser \
    --output my_baseline
```

### 查询数据

```bash
# 查询特定配置
python -m python.tools.ggml_baseline_parser \
    --query \
    --hardware RTX4090 \
    --type q4_0 \
    --m 4096 \
    --n 1 \
    --k 4096
```

---

## 数据覆盖

### 硬件

| 硬件 | 记录数 |
|------|--------|
| RTX4090 | 588 |
| A100 | 582 |
| RTX4070 | 504 |

### 量化类型

| 类型 | 说明 |
|------|------|
| q4_0 | 4-bit 对称量化 |
| q4_1 | 4-bit 非对称量化 |
| q8_0 | 8-bit 对称量化 |

### 维度范围

| 维度 | 范围 |
|------|------|
| M | 512 ~ 152064 |
| N | 1 ~ 512 |
| K | 512 ~ 18944 |

---

## 在 Benchmark 中使用

### 示例代码

```python
from core.tools.ggml_baseline_api import BaselineAPI

class BenchmarkRunner:
    def __init__(self, hardware="RTX4090"):
        self.api = BaselineAPI()
        self.hardware = hardware

    def run_test(self, definition_path: str):
        # 运行测试
        result = self.execute_test(definition_path)

        # 获取基线
        m, n, k = result["m"], result["n"], result["k"]
        type_a = result["quant_type"]

        baseline = self.api.get_baseline(self.hardware, type_a, m, n, k)

        if baseline:
            comparison = self.api.compare_performance(
                self.hardware, type_a, m, n, k, result["tflops"]
            )

            print(f"\n{'='*50}")
            print(f"性能对比 ({self.hardware})")
            print(f"{'='*50}")
            print(f"配置: {type_a.upper()}, M={m}, N={n}, K={k}")
            print(f"基线: {baseline['tflops']:.2f} TFLOPS")
            print(f"实际: {result['tflops']:.2f} TFLOPS")
            print(f"比率: {comparison['ratio']:.2%}")

            if comparison['better']:
                print(f"✅ 比基线快 {comparison['diff_percent']:.1%}")
            else:
                print(f"❌ 比基线慢 {-comparison['diff_percent']:.1%}")
        else:
            print(f"未找到基线数据，尝试最接近的配置...")
            closest = self.api.get_closest_baseline(
                self.hardware, type_a, m, n, k
            )
            if closest:
                print(f"最接近配置: M={closest['m']}, N={closest['n']}, K={closest['k']}")
                print(f"基线性能: {closest['tflops']:.2f} TFLOPS")
```

---

## 常见用例

### 1. 批量测试比较

```python
api = BaselineAPI()

test_results = [
    {"type": "q4_0", "m": 4096, "n": 1, "k": 4096, "tflops": 5.2},
    {"type": "q4_0", "m": 14336, "n": 1, "k": 4096, "tflops": 8.5},
    {"type": "q4_1", "m": 4096, "n": 1, "k": 4096, "tflops": 5.5},
]

for result in test_results:
    comparison = api.compare_performance(
        "RTX4090", result["type"], result["m"], result["n"], result["k"],
        result["tflops"]
    )
    print(f"{result['type']}: {comparison['ratio']:.2%} vs baseline")
```

### 2. 查找最佳配置

```python
api = BaselineAPI()

# 查找某个硬件上特定类型的最佳性能
hardware = "RTX4090"
type_a = "q4_0"

best_tflops = 0
best_config = None

for case_id, case_data in api.data.items():
    if case_data["type_a"] != type_a:
        continue

    hw_data = case_data["hardware"].get(hardware)
    if hw_data and hw_data["tflops"] > best_tflops:
        best_tflops = hw_data["tflops"]
        best_config = case_data

if best_config:
    print(f"最佳性能: {best_tflops} TFLOPS")
    print(f"配置: M={best_config['m']}, N={best_config['n']}, K={best_config['k']}")
```

---

## 注意事项

1. **数据更新**: 当 ggml-python-main 结果更新时，需要重新运行解析器
2. **硬件名称**: 使用标准名称 (A100, RTX4090, RTX4070)
3. **维度匹配**: 如果维度不完全匹配，使用 `get_closest_baseline()`
4. **性能单位**: 统一使用 TFLOPS，可通过 `gflops` 获取 GFLOPS

---

## 故障排除

### 找不到基线数据

```python
# 检查可用硬件
api = BaselineAPI()
print(api.get_all_hardware())  # ['A100', 'RTX4070', 'RTX4090']

# 检查可用 case_id (新格式: w{weight}a{activation}c{compute}_...)
print(api.case_id_list[:10])
# ['w4a32c8_q4_0_f32_m4096_n1_k4096', 'w4a32c8_q4_0_f32_m14336_n1_k4096', ...]

# 使用最接近的配置
closest = api.get_closest_baseline("RTX4090", "q4_0", 4096, 2, 4096)
```

### 重新生成数据

```bash
cd /home/haiyan/Agent4Kernel/KernelEvalPlus
python -m python.tools.ggml_baseline_parser \
    --results-dir /home/haiyan/Agent4Kernel/ggml-python-main/results
```
