# Baseline 系统架构与检索逻辑详解

## 概述

Baseline 系统用于存储和查询 GGML 框架的性能基准数据，支持以下算子类型：

| 算子类型 | 性能指标 | 数据文件 |
|---------|---------|---------|
| Quantized GEMM | TFLOPS | `baseline_data_compact.json` (496KB) |
| Flash Attention | TFLOPS | `flash_attn_baseline.json` (60KB) |
| RMS Norm | GB/s | `rms_norm_baseline.json` (24KB) |
| TopK | GB/s | `topk_baseline.json` (12KB) |

---

## 1. 数据存储结构

### 1.1 存储位置

```
data/baseline/
├── baseline_data_compact.json  # Quantized GEMM baseline
├── flash_attn_baseline.json    # Flash Attention baseline
├── rms_norm_baseline.json      # RMS Norm baseline
└── topk_baseline.json          # TopK baseline
```

### 1.2 JSON 数据结构

#### RMS Norm 示例

```json
{
  "rms_norm_hs1536_1536x512x1x1": {
    "hidden_size": 1536,
    "ne": [1536, 512, 1, 1],
    "hardware": {
      "RTX4090": {"gbps": 1064.63, "us_per_run": 5.5},
      "A100": {"gbps": 660.35, "us_per_run": 8.87},
      "RTX4070": {"gbps": 287.97, "us_per_run": 20.35}
    }
  }
}
```

#### GEMM 示例

```json
{
  "w4a32c8_q4_0_f32_m4096_n1_k4096": {
    "type_a": "q4_0",
    "type_b": "f32",
    "m": 4096,
    "n": 1,
    "k": 4096,
    "hardware": {
      "RTX4090": {"tflops": 4.95, "gflops": 4950.0, "us_per_run": 6.77}
    }
  }
}
```

#### Flash Attention 示例

```json
{
  "flash_attn_Llama3-8B_F16_cache512_nb512": {
    "model": "Llama3-8B",
    "kv_type": "F16",
    "kv_cache_size": 512,
    "nb": 512,
    "hardware": {
      "RTX4090": {"tflops": 6.42, "us_per_run": 12.5}
    }
  }
}
```

### 1.3 case_id 命名规则

| 算子 | case_id 格式 | 示例 |
|-----|-------------|------|
| GEMM | `w{w_bits}a32c8_{quant_type}_{act_type}_m{M}_n{N}_k{K}` | `w4a32c8_q4_0_f32_m4096_n1_k4096` |
| RMS Norm | `rms_norm_hs{hidden_size}_{ne0}x{ne1}x{ne2}x{ne3}` | `rms_norm_hs1536_1536x512x1x1` |
| Flash Attention | `flash_attn_{model}_{kv_type}_cache{cache_size}_nb{batch_size}` | `flash_attn_Llama3-8B_F16_cache512_nb512` |
| TopK | `topk_k{k}_ne0{ne0}_{ne0}x{ne1}x{ne2}x{ne3}` | `topk_k8_ne0256_256x512x1x1` |

---

## 2. 核心组件

### 2.1 组件架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         测试请求入口                                      │
│            unified_test_runner.py --test --definition xxx.json          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        UnifiedTestRunner                                 │
│  - 加载 definition JSON                                                  │
│  - 检测算子类型 (detect_op_type)                                         │
│  - 获取对应的 Handler                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     OperatorTestHandler (基类)                           │
│  子类: QuantGEMMHandler, FlashAttentionHandler,                         │
│        RMSNormHandler, TopKHandler                                       │
│                                                                          │
│  核心方法:                                                               │
│  - generate_inputs()     生成测试输入                                    │
│  - get_reference_output() 参考实现                                       │
│  - run_kernel()          运行 kernel                                     │
│  - calculate_performance() 计算性能指标                                  │
│  - query_baseline()      查询 baseline ← 关键方法                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          BaselineAPI                                     │
│  - 加载所有 JSON baseline 文件                                           │
│  - 提供 get_gemm(), get_flash_attn(), get_rms_norm(), get_topk() 方法   │
│  - 根据 case_id 和 hardware 查询性能数据                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       data/baseline/*.json                               │
│  - baseline_data_compact.json (GEMM)                                    │
│  - flash_attn_baseline.json                                             │
│  - rms_norm_baseline.json                                               │
│  - topk_baseline.json                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 文件位置

| 组件 | 文件路径 |
|-----|---------|
| BaselineAPI | `core/tools/baseline_api.py` |
| OperatorTestHandler | `llm_kernel_test/op_test_handler.py` |
| UnifiedTestRunner | `llm_kernel_test/unified_test_runner.py` |
| 数据文件 | `data/baseline/*.json` |

---

## 3. 检索流程详解

### 3.1 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 1: 测试请求                                                        │
│                                                                         │
│ python llm_kernel_test/unified_test_runner.py --test \                  │
│     --definition definitions/rms_norm/qwen/fp32_rms_norm_qwen3_4b_hs1536.json │
│     --attempt-path attempts/rms_norm_v1                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 2: 加载 Definition                                                 │
│                                                                         │
│ spec = {                                                                │
│   "name": "fp32_rms_norm_qwen3_4b_hs1536",                              │
│   "op_type": "rms_norm",                                                │
│   "axes": {"hidden_size": {"value": 1536}, "epsilon": {"value": 1e-6}}, │
│   "test_configs": [                                                     │
│     {"name": "batch_1", "batch_size": 1},                               │
│     {"name": "batch_8", "batch_size": 8},                               │
│     {"name": "batch_512", "batch_size": 512}                            │
│   ]                                                                     │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 3: 获取 Handler                                                    │
│                                                                         │
│ op_type = detect_op_type(spec)  # "rms_norm"                            │
│ handler = get_handler(op_type)  # RMSNormHandler                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 4: 正确性测试 (_test_correctness)                                   │
│                                                                         │
│ for test_config in test_configs:  # 3 个配置                            │
│     inputs = handler.generate_inputs(spec, test_config, device)         │
│     output = handler.run_kernel(kernel_func, inputs, spec)              │
│     ref_output = handler.get_reference_output(spec, inputs)             │
│     # 计算 NMSE 验证正确性                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 5: 性能测试 (_test_performance)                                     │
│                                                                         │
│ for test_config in test_configs:  # 3 个配置                            │
│     # Warmup 10 次 + Benchmark 100 次                                   │
│     perf = handler.calculate_performance(output, latency_ms, ...)       │
│                                                                          │
│     # Baseline 对比                                                     │
│     baseline = handler.query_baseline(spec, hardware, test_config)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 6: Handler.query_baseline() 详细流程                               │
│                                                                         │
│ # RMSNormHandler.query_baseline() (op_test_handler.py:691-716)          │
│ def query_baseline(self, spec, hardware, test_config):                  │
│     from core.tools.baseline_api import BaselineAPI                   │
│     api = BaselineAPI()  # ⚠️ 每次调用都创建新实例                       │
│                                                                         │
│     hidden_size = spec["axes"]["hidden_size"]["value"]  # 1536          │
│     batch_size = test_config.get("batch_size", 512)    # 1/8/512        │
│     ne = [hidden_size, 1, batch_size, 1]  # [1536, 1, 1, 1]             │
│                                                                         │
│     baseline = api.get_rms_norm(hardware, hidden_size, ne)              │
│     return baseline                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 7: BaselineAPI 初始化 (baseline_api.py:62-88)                       │
│                                                                         │
│ def __init__(self, baseline_dir=None):                                  │
│     self.baseline_dir = Path(baseline_dir)                              │
│     self._data = {}                                                     │
│     self._load_all_data()  # 加载所有 4 个 JSON 文件 (~600KB)           │
│                                                                         │
│ def _load_all_data(self):                                               │
│     for op_type, filename in self.OPERATOR_FILES.items():               │
│         self._data[op_type] = self._load_json(filename)                 │
│     # 加载: baseline_data_compact.json (496KB)                          │
│     #      flash_attn_baseline.json (60KB)                              │
│     #      rms_norm_baseline.json (24KB)                                │
│     #      topk_baseline.json (12KB)                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 8: BaselineAPI.get_rms_norm() 查询 (baseline_api.py:226-253)        │
│                                                                         │
│ def get_rms_norm(self, hardware, hidden_size, ne):                      │
│     ne_str = "x".join(map(str, ne))  # "1536x1x1x1"                     │
│     case_id = f"rms_norm_hs{hidden_size}_{ne_str}"                      │
│     # case_id = "rms_norm_hs1536_1536x1x1x1"                            │
│                                                                         │
│     data = self._data["rms_norm"].get(case_id, {})                      │
│     hw_data = data.get("hardware", {}).get(hardware)                    │
│                                                                         │
│     if hw_data:                                                         │
│         return {"gbps": hw_data.get("gbps"), "us_per_run": ...}         │
│     return None                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 各 Handler 的 case_id 构建逻辑

#### RMSNormHandler

```python
# op_test_handler.py:691-716
def query_baseline(self, spec, hardware, test_config):
    hidden_size = spec["axes"]["hidden_size"]["value"]  # 从 definition 获取
    batch_size = test_config.get("batch_size", 512)     # 从 test_config 获取
    ne = [hidden_size, 1, batch_size, 1]

    # case_id = "rms_norm_hs1536_1536x1x1x1"
    case_id = f"rms_norm_hs{hidden_size}_{'x'.join(map(str, ne))}"
```

#### QuantGEMMHandler

```python
# op_test_handler.py:268-294
def query_baseline(self, spec, hardware, test_config):
    M, N, K = self._get_dims(spec, test_config)
    weight_dtype = spec.get("inputs", {}).get("weight", {}).get("dtype", "block_q8_0")
    type_a = weight_dtype.replace("block_", "")  # "q8_0"

    # case_id = "w8a32c8_q8_0_f32_m4096_n1_k4096"
    # 注意: GGML 维度映射 M<->N
    baseline = api.get_gemm(hardware, type_a, N, M, K)
```

#### FlashAttentionHandler

```python
# op_test_handler.py:599-638
def query_baseline(self, spec, hardware, test_config):
    model = spec.get("model_architectures", ["unknown"])[0]
    kv_type = "F16"  # 从 tags 提取
    cache_size = spec["axes"]["seq_len"]["value"]
    batch_size = test_config.get("batch_size", 512)

    # case_id = "flash_attn_Llama3-8B_F16_cache512_nb512"
    baseline = api.get_flash_attn(hardware, model, kv_type, cache_size, batch_size)
```

#### TopKHandler

```python
# op_test_handler.py:781-809
def query_baseline(self, spec, hardware, test_config):
    k = spec["axes"]["k"]["value"]
    vocab_subset = 256
    batch_size = test_config.get("batch_size", 512)
    ne = [vocab_subset, batch_size, 1, 1]

    # case_id = "topk_k8_ne0256_256x512x1x1"
    baseline = api.get_topk(hardware, k, vocab_subset, ne)
```

---

## 4. 性能问题分析

### 4.1 问题定位

**核心问题**: 每次 `query_baseline()` 调用都会创建新的 `BaselineAPI()` 实例

```python
# op_test_handler.py 中各 Handler 的 query_baseline 方法
def query_baseline(self, spec, hardware, test_config):
    try:
        from core.tools.baseline_api import BaselineAPI
        api = BaselineAPI()  # ⚠️ 每次都创建新实例
        ...
```

### 4.2 性能影响估算

| 场景 | 计算 | 结果 |
|-----|------|------|
| 单次测试 | 3 个 test_config × 1 次 BaselineAPI() | 3 次 JSON 加载 |
| 批量测试 100 个 attempt | 100 × 3 × 1 | 300 次 JSON 加载 |
| JSON 文件总大小 | 496K + 60K + 24K + 12K | ~600KB |
| 总 I/O 量 | 300 × 600KB | ~180MB |

### 4.3 问题代码位置

```python
# llm_kernel_test/op_test_handler.py

class QuantGEMMHandler(OperatorTestHandler):
    def query_baseline(self, spec, hardware, test_config):
        api = BaselineAPI()  # 第 272 行

class FlashAttentionHandler(OperatorTestHandler):
    def query_baseline(self, spec, hardware, test_config):
        api = BaselineAPI()  # 第 603 行

class RMSNormHandler(OperatorTestHandler):
    def query_baseline(self, spec, hardware, test_config):
        api = BaselineAPI()  # 第 695 行

class TopKHandler(OperatorTestHandler):
    def query_baseline(self, spec, hardware, test_config):
        api = BaselineAPI()  # 第 785 行
```

---

## 5. 优化建议

### 5.1 方案一：模块级缓存 (推荐)

修改 `core/tools/baseline_api.py`：

```python
# 模块级缓存
_cached_api = None

def get_baseline_api() -> BaselineAPI:
    """获取单例 BaselineAPI 实例"""
    global _cached_api
    if _cached_api is None:
        _cached_api = BaselineAPI()
    return _cached_api
```

修改 `llm_kernel_test/op_test_handler.py`：

```python
# 修改前
from core.tools.baseline_api import BaselineAPI
api = BaselineAPI()

# 修改后
from core.tools.baseline_api import get_baseline_api
api = get_baseline_api()
```

### 5.2 方案二：在 UnifiedTestRunner 中缓存

修改 `llm_kernel_test/unified_test_runner.py`：

```python
class UnifiedTestRunner:
    def __init__(self, config_path="..."):
        ...
        self.baseline_api = BaselineAPI()  # 初始化时创建一次

    def _test_performance(self, ...):
        # 传递 api 给 handler
        baseline = handler.query_baseline(spec, hardware, test_config, self.baseline_api)
```

### 5.3 预期效果

| 优化前 | 优化后 |
|-------|-------|
| 每个 test_config 加载 4 个 JSON | 整个测试过程只加载 1 次 |
| 300 次 JSON 解析 (100 attempts) | 1 次 JSON 解析 |
| ~180MB I/O | ~600KB I/O |

---

## 6. 相关文档

- [GGML Baseline 使用指南](./ggml_baseline_usage.md) - GEMM baseline API 使用方法
- [新算子 Baseline 系统使用指南](../../core/tools/NEW_OPERATORS_BASELINE_README.md) - Flash Attention、RMS Norm、TopK API 使用方法
- [统一测试框架](./llm_kernel_testing.md) - 测试框架整体设计
