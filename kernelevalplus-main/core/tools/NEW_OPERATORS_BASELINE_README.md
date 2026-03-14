# 新算子 Baseline 系统使用指南

## 概述

本系统扩展了原有的 GEMM baseline 系统，支持以下新算子的基线数据查询：

- **Flash Attention**: 基于 TFLOPS 性能指标
- **RMS Norm**: 基于 GB/s 内存带宽指标
- **TopK**: 基于 GB/s 内存带宽指标

## 文件结构

```
kernelevalplus/
├── data/baseline/
│   ├── baseline_data_compact.json      # GEMM baseline (原有)
│   └── new_operators_baseline.json     # 新算子 baseline (新增)
├── core/tools/
│   ├── add_a100_new_operators_baseline.py    # 解析新算子数据脚本
│   └── new_operators_baseline_api.py         # 新算子查询 API
└── bench_web/static_site/data/
    └── new_operators_baseline.json           # Web 端数据 (自动同步)
```

## 数据格式

### Flash Attention

```json
{
  "flash_attn": {
    "flash_attn_Llama3-8B_F16_cache512_nb1": {
      "model": "Llama3-8B",
      "kv_type": "F16",
      "kv_cache_size": 512,
      "nb": 1,
      "hardware": {
        "A100": {
          "tflops": 0.42,
          "us_per_run": 20.04
        }
      }
    }
  }
}
```

### RMS Norm

```json
{
  "rms_norm": {
    "rms_norm_hs4096_4096x512x1x1": {
      "hidden_size": 4096,
      "ne": [4096, 512, 1, 1],
      "hardware": {
        "A100": {
          "gbps": 1566.07,
          "us_per_run": 9.98
        }
      }
    }
  }
}
```

### TopK

```json
{
  "topk": {
    "topk_k8_ne0_256_256x512x1x1": {
      "k": 8,
      "ne0": 256,
      "ne": [256, 512, 1, 1],
      "hardware": {
        "A100": {
          "gbps": 20.42,
          "us_per_run": 24.67
        }
      }
    }
  }
}
```

## API 使用方法

### 1. 导入 API

```python
from core.tools.new_operators_baseline_api import NewOperatorsBaselineAPI
```

### 2. Flash Attention 查询

```python
api = NewOperatorsBaselineAPI()

# 精确查询
baseline = api.get_flash_attn(
    hardware="A100",
    model="Llama3-8B",
    kv_type="F16",
    kv_cache_size=512,
    nb=512
)

if baseline:
    print(f"TFLOPS: {baseline['tflops']}")
    print(f"延迟: {baseline['us_per_run']} us")

# 按模型查询所有配置
results = api.get_flash_attn_by_model("A100", "Llama3-8B")
for r in results:
    print(f"{r['kv_type']} cache={r['kv_cache_size']} nb={r['nb']}: {r['tflops']} TFLOPS")
```

### 3. RMS Norm 查询

```python
api = NewOperatorsBaselineAPI()

# 精确查询
baseline = api.get_rms_norm(
    hardware="A100",
    hidden_size=4096,
    ne=[4096, 512, 1, 1]
)

if baseline:
    print(f"GB/s: {baseline['gbps']}")
    print(f"延迟: {baseline['us_per_run']} us")

# 按 hidden_size 查询所有配置
results = api.get_rms_norm_by_hidden_size("A100", 4096)
for r in results:
    print(f"ne={r['ne']}: {r['gbps']} GB/s")
```

### 4. TopK 查询

```python
api = NewOperatorsBaselineAPI()

# 精确查询
baseline = api.get_topk(
    hardware="A100",
    k=8,
    ne0=256,
    ne=[256, 512, 1, 1]
)

if baseline:
    print(f"GB/s: {baseline['gbps']}")
    print(f"延迟: {baseline['us_per_run']} us")

# 按 k 值查询所有配置
results = api.get_topk_by_k("A100", 8)
for r in results:
    print(f"ne0={r['ne0']} ne={r['ne']}: {r['gbps']} GB/s")
```

### 5. 快捷函数

```python
from core.tools.new_operators_baseline_api import (
    get_flash_attn_baseline,
    get_rms_norm_baseline,
    get_topk_baseline
)

# 快捷查询
baseline = get_flash_attn_baseline("A100", "Llama3-8B", "F16", 512, 512)
baseline = get_rms_norm_baseline("A100", 4096, [4096, 512, 1, 1])
baseline = get_topk_baseline("A100", 8, 256, [256, 512, 1, 1])
```

## 添加新硬件数据

如果需要为其他硬件添加新算子的 baseline 数据：

1. 运行 GGML 测试获取新硬件的数据
2. 将 markdown 格式的结果放到 `/home/qinhaiyan/ggml-python/results/{hardware-name}/`
3. 修改 `add_a100_new_operators_baseline.py` 脚本，添加新硬件的处理逻辑
4. 运行脚本更新 baseline 数据

```bash
python -m python.tools.add_a100_new_operators_baseline
```

## 更新 Web 数据

更新 baseline 数据后，需要同步到 web 端：

```bash
cd bench_web/static_site
python generate_static.py
```

## 当前数据统计

- **Flash Attention**: 126 entries
  - 模型: Llama3-8B, Qwen2.5-7B, DeepSeekV2 (不支持)
  - KV 类型: F16, Q8_0, Q4_0
  - Cache 大小: 512, 4096, 8192
  - Block 数量: 1, 2, 3, 4, 5, 8, 512

- **RMS Norm**: 63 entries
  - Hidden Size: 128, 512, 1536, 2560, 3584, 4096, 5120, 7168
  - 形状: [hidden_size, 1/8/32, 1-512, 1]

- **TopK**: 28 entries
  - K 值: 6, 8
  - ne0: 160, 256
  - 形状: [ne0, 1-512, 1, 1]

## 性能参考 (A100)

| 算子 | 最佳配置 | 峰值性能 |
|------|----------|----------|
| Flash Attention | Llama3-8B, F16, cache=8192, nb=512 | 113.29 TFLOPS |
| RMS Norm | hidden_size=4096, ne=[4096,512,1,1] | 1566.07 GB/s |
| TopK | k=8, ne0=256, ne=[256,512,1,1] | 20.42 GB/s |
