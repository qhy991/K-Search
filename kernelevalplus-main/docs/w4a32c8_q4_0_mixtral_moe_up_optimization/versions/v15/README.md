# v15 - 最优版本

## 配置
- **小批次 (M ≤ 8)**: 64线程 + 8-way Split-K
- **大批次 (M > 8)**: 256线程, 无Split-K

## 性能
| Batch | GFLOPS | Latency |
|-------|--------|---------|
| M=1 | 1152.3 | 0.102 ms |
| M=2 | 1041.6 | 0.226 ms |
| M=512 | 1169.1 | 51.435 ms |

## Baseline对比
- 性能比率: **14.0%** of GGML baseline
- 相比v1提升: **146.5%**

## 使用方法
```python
import torch

# 加载编译后的模块
module = torch.ops.load_library("kernel.so")

# 调用
output = module.forward(weight, activation, M, N, K)
```
