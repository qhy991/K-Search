# W4A32C8 Q4_1 Kernel 实验保存位置完整指南

## 🎯 快速定位

### 最重要的文件 ⭐⭐⭐

| 文件 | 位置 | 说明 |
|------|------|------|
| **最终测试结果** | `llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json` | 包含所有性能数据 |
| **最终优化kernel** | `llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/kernel.cu` | 最终优化版本代码 |
| **优化历程文档** | `docs/q4_1_optimization_journey.md` | 完整开发过程 |
| **模板kernel** | `llm_kernel_test/templates/w4a32c8_q4_1_fp32_int8/kernel.cu` | 生产环境使用的代码 |

---

## 📁 完整目录结构

```
/home/qinhaiyan/kernelevalplus/
│
├── llm_kernel_test/
│   ├── templates/w4a32c8_q4_1_fp32_int8/        # 🔧 模板代码（源代码）
│   │   ├── kernel.cu                             # 优化版本（默认）
│   │   ├── kernel_basic.cu                       # 基础版本
│   │   ├── kernel_optimized.cu                   # 优化版本源文件
│   │   ├── bindings.cpp                          # PyTorch绑定
│   │   ├── impl.json                             # 元数据
│   │   └── reference.py                          # Python参考
│   │
│   └── sandbox/generated/                        # 🧪 实验结果
│       ├── v1/w4a32c8_q4_1_fp32_int8/            # 基础版本实验
│       │   ├── kernel.cu
│       │   ├── test_results.json                 # 基础版本性能
│       │   └── metadata.json
│       │
│       ├── v2_optimized/w4a32c8_q4_1_fp32_int8/  # 中间优化版本
│       │   ├── kernel.cu
│       │   └── test_results.json
│       │
│       ├── v2_optimized_final/w4a32c8_q4_1_fp32_int8/  # ⭐ 最终版本
│       │   ├── kernel.cu                         # 最终优化代码
│       │   ├── kernel_basic.cu                   # 基础版本
│       │   ├── kernel_optimized.cu               # 优化版本
│       │   ├── test_results.json                 # ⭐ 最终测试结果
│       │   ├── metadata.json
│       │   └── reference.py
│       │
│       └── v1_basic/w4a32c8_q4_1_fp32_int8/      # 基础版本备份
│           └── test_results.json
│
├── docs/                                         # 📚 文档
│   ├── INDEX.md                                  # 文档索引
│   ├── Q4_1_OPTIMIZATION_README.md               # 项目README
│   ├── w4a32c8_q4_1_implementation_complete.md   # 实现报告
│   ├── q4_1_optimization_journey.md              # ⭐ 优化历程（最重要）
│   ├── q4_1_optimization_report.md               # 优化报告
│   └── q4_1_optimization_quick_reference.md      # 快速参考
│
├── definitions/quant_gemm/deepseek_v2/           # 📋 定义文件
│   └── w4a32c8_q4_1_fp32_int8_deepseek_v2_att_out_n5120_k5120.json
│
└── [根目录]                                      # 🧪 测试脚本
    ├── test_q4_1_kernel.py                       # 基础测试
    ├── test_q4_1_kernel_optimized.py             # 优化版本测试
    └── compare_q4_1_kernels.py                   # 性能对比
```

---

## 📊 查看实验结果

### 1. 查看最终测试结果

```bash
# 完整JSON
cat llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json

# 或使用jq格式化查看
cat llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json | jq '.'

# 只看性能数据
cat llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json | jq '.performance'
```

### 2. 查看所有版本对比

```bash
# v1基础版本
cat llm_kernel_test/sandbox/generated/v1/w4a32c8_q4_1_fp32_int8/test_results.json | jq '.performance'

# v2最终版本
cat llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json | jq '.performance'
```

### 3. 列出所有实验

```bash
ls -la llm_kernel_test/sandbox/generated/
```

---

## 🔍 按需求查找

### 需要源代码
→ **llm_kernel_test/templates/w4a32c8_q4_1_fp32_int8/**

### 需要测试结果
→ **llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json**

### 需要对比数据
→ 查看 **docs/q4_1_optimization_quick_reference.md** (已汇总)

### 需要理解优化过程
→ 阅读 **docs/q4_1_optimization_journey.md**

### 需要快速上手
→ 阅读 **docs/Q4_1_OPTIMIZATION_README.md**

---

## 📈 测试结果文件内容

### test_results.json 结构

```json
{
  "attempt_id": "v2_optimized_final",
  "variant": "w4a32c8_q4_1_fp32_int8",
  "tested_at": "2026-02-12T22:23:xx",

  "compilation": {
    "success": true,
    "errors": [],
    "warnings": []
  },

  "correctness": {
    "passed": true,
    "nmse": 0.0,
    "test_cases": [
      {"shape": "batch_1", "passed": true, "nmse": 0.000000},
      {"shape": "batch_2", "passed": true, "nmse": 0.000001},
      ...
      {"shape": "batch_512", "passed": true, "nmse": 0.000000}
    ]
  },

  "performance": {
    "test_cases": {
      "batch_1": {
        "M": 1, "N": 5120, "K": 5120,
        "latency_ms": 0.031,
        "gflops": 1705.5
      },
      "batch_512": {
        "M": 512, "N": 5120, "K": 5120,
        "latency_ms": 1.867,
        "gflops": 14375.6
      },
      ...
    }
  }
}
```

---

## 🚀 重新运行实验

### 测试最终版本

```bash
cd /home/qinhaiyan/kernelevalplus

# 使用默认优化kernel测试
python test_q4_1_kernel.py

# 查看结果
cat llm_kernel_test/sandbox/generated/v1/w4a32c8_q4_1_fp32_int8/test_results.json | jq '.performance'
```

### 对比基础版本vs优化版本

```bash
# 注意：需要修改compare_q4_1_kernels.py才能正常运行
python compare_q4_1_kernels.py
```

### 只测试优化版本

```bash
python test_q4_1_kernel_optimized.py
```

---

## 💾 备份和恢复

### 备份实验结果

```bash
# 备份整个sandbox目录
tar -czf q4_1_experiments_$(date +%Y%m%d).tar.gz \
    llm_kernel_test/sandbox/generated/*q4_1* \
    llm_kernel_test/sandbox/generated/v* \
    docs/q4_1*.md

# 只备份测试结果
tar -czf q4_1_results_$(date +%Y%m%d).tar.gz \
    llm_kernel_test/sandbox/generated/*/w4a32c8_q4_1_fp32_int8/test_results.json
```

### 恢复实验

```bash
# 解压备份
tar -xzf q4_1_experiments_20260212.tar.gz
```

---

## 📝 实验元数据

### 查看实验元数据

```bash
# v2_optimized_final的元数据
cat llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/metadata.json

# 包含:
# - 创建时间
# - Git commit hash
# - 测试配置
# - GPU信息
```

---

## 🔗 相关文件链接

### Git历史
```bash
# 查看相关commits
git log --oneline --grep="Q4_1\|q4_1" -10

# 查看最近的Q4_1相关更改
git log --all --oneline -- "*q4_1*" -5
```

### 编译产物
```bash
# 编译生成的.so文件
ls -lh llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/*.so
```

---

## 📞 快速命令参考

```bash
# 工作目录
cd /home/qinhaiyan/kernelevalplus

# 查看最终性能
cat llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json | jq '.performance.test_cases | to_entries[] | {name: .key, gflops: .value.gflops}'

# 查看所有实验版本
ls -d llm_kernel_test/sandbox/generated/*/ | grep -E "v[0-9]|optimized"

# 查看文档列表
ls docs/*q4_1*

# 搜索实验文件
find . -name "*test_results.json" -path "*/q4_1*" -o -name "*test_results.json" -path "*/v2_optimized*"
```

---

## 🎯 总结

### 最重要的3个位置

1. **测试结果**: `llm_kernel_test/sandbox/generated/v2_optimized_final/w4a32c8_q4_1_fp32_int8/test_results.json`
2. **优化代码**: `llm_kernel_test/templates/w4a32c8_q4_1_fp32_int8/kernel.cu`
3. **优化历程**: `docs/q4_1_optimization_journey.md`

### 版本说明

- **v1**: 基础版本（M=8性能悬崖）
- **v2_optimized**: 中间优化版本
- **v2_optimized_final**: 最终优化版本（推荐使用）⭐
- **v1_basic**: 基础版本备份

---

**最后更新**: 2026-02-12
**实验状态**: ✅ 完成
**数据完整性**: ✅ 已验证
