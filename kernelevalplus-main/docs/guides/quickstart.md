# KernelEvalPlus 快速开始指南

## 环境准备

### 1. 检查 CUDA 环境

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU
nvidia-smi
```

### 2. 创建 Python 环境（推荐）

```bash
# 使用 conda
conda create -n kerneleval python=3.10
conda activate kerneleval

# 或使用 venv
python3 -m venv ~/kerneleval_env
source ~/kerneleval_env/bin/activate
```

### 3. 安装 PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 安装 KernelEvalPlus

### 方式1: 开发模式安装（推荐）

```bash
cd ~/Agent4Kernel/KernelEvalPlus/python
pip install -e .
```

开发模式的优点：
- 代码修改立即生效
- 便于调试和开发
- 适合测试和实验

### 方式2: 正常安装

```bash
cd ~/Agent4Kernel/KernelEvalPlus/python
pip install .
```

### 指定 CUDA 架构（可选）

如果编译时遇到问题，可以指定目标 GPU 架构：

```bash
# RTX 3090 (Ampere, SM 8.6)
TORCH_CUDA_ARCH_LIST="8.6" pip install -e .

# RTX 4090 (Ada Lovelace, SM 8.9)
TORCH_CUDA_ARCH_LIST="8.9" pip install -e .

# A100 (Ampere, SM 8.0)
TORCH_CUDA_ARCH_LIST="8.0" pip install -e .

# 多个架构
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install -e .
```

## 验证安装

### 1. 基础验证

```bash
cd ~/Agent4Kernel/KernelEvalPlus/python
python3 validate_installation.py
```

### 2. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_gemm_generic.py -v

# 运行正确性测试
pytest tests/ -v -k "correctness"

# 运行性能测试
pytest tests/ -v -k "performance"
```

### 3. 运行 Benchmark

```bash
# 使用算子框架
python3 test_operators_framework.py

# 使用测试框架演示
python3 test_framework_demo.py
```

## 基本使用

### 示例 1: 基础量化和 GEMM

```python
import torch
import quant_gemm

# 设置参数
M, N, K = 4096, 2, 4096
device = 'cuda'

# 创建测试数据
weight = torch.randn(M, K, device=device, dtype=torch.float32)
activation = torch.randn(N, K, device=device, dtype=torch.float32)

# 量化
weight_q = quant_gemm.quantize_q4_0(weight)       # [M, K] -> [M, K//32, 18]
activation_q = quant_gemm.quantize_q8_1(activation)  # [N, K] -> [N, K//32, 36]

# 运行量化 GEMM
output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

# 对比 FP32 参考实现
output_ref = weight @ activation.T
error = torch.mean((output - output_ref)**2) / torch.mean(output_ref**2)
print(f"NMSE: {error.item():.6e}")
```

### 示例 2: 使用算子框架

```python
from operators.registry import OperatorRegistry

# 创建注册表
registry = OperatorRegistry()

# 列出所有可用算子
print("Available operators:")
for family, variants in registry.list_operators().items():
    print(f"  {family}:")
    for variant in variants:
        print(f"    - {variant}")

# 获取特定算子
op = registry.get_operator("quant_gemm", "w4a16_q4_0_fp32")

# 运行正确性测试
print("\nRunning correctness test...")
op.test_correctness(M=4096, N=2, K=4096)

# 运行性能测试
print("\nRunning benchmark...")
op.benchmark(M=4096, N=2, K=4096, warmup=10, iterations=100)
```

### 示例 3: 自定义测试

```python
from operators.test_framework import TestFramework

# 创建测试框架
framework = TestFramework()

# 添加测试配置
test_config = {
    "operator": "quant_gemm.w4a16_q4_0_fp32",
    "shapes": [
        {"M": 4096, "N": 1, "K": 4096},
        {"M": 4096, "N": 2, "K": 4096},
        {"M": 4096, "N": 4, "K": 4096},
    ],
    "correctness_threshold": 1e-2,
    "benchmark_iterations": 100,
}

# 运行测试
results = framework.run_test(test_config)

# 打印结果
framework.print_results(results)
```

## 常见问题

### Q1: 编译失败，提示找不到 CUDA

**解决方案**:
```bash
# 检查 CUDA 路径
echo $CUDA_HOME
echo $PATH

# 设置 CUDA 路径（如果未设置）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Q2: 编译失败，提示架构不匹配

**解决方案**:
```bash
# 检查你的 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv

# 指定正确的架构
TORCH_CUDA_ARCH_LIST="8.6" pip install -e .
```

### Q3: 运行时错误，提示找不到 _C 模块

**解决方案**:
```bash
# 重新编译
cd ~/Agent4Kernel/KernelEvalPlus/python
pip uninstall quant_gemm -y
pip install -e .
```

### Q4: 测试失败，精度不符合预期

**解决方案**:
- 检查输入数据范围
- 调整正确性阈值
- 查看具体的错误信息

### Q5: 性能不如预期

**解决方案**:
- 确保 GPU 处于性能模式
- 增加 warmup 迭代次数
- 检查是否有其他进程占用 GPU

## 性能优化建议

### 1. GPU 设置

```bash
# 设置性能模式
sudo nvidia-smi -pm 1

# 设置最大时钟频率
sudo nvidia-smi -lgc 1410  # 根据你的 GPU 调整
```

### 2. 编译优化

```bash
# 使用 O3 优化（默认已启用）
# 使用 fast math（默认已启用）
# 针对特定架构编译
TORCH_CUDA_ARCH_LIST="8.9" pip install -e .
```

### 3. 运行时优化

```python
# 使用 torch.cuda.synchronize() 确保准确计时
import torch

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# 运行 kernel
end.record()
torch.cuda.synchronize()

elapsed_time = start.elapsed_time(end)  # 毫秒
```

## 下一步

1. **探索示例**: 查看 `examples/` 目录中的示例代码
2. **阅读文档**:
   - `KERNEL_IMPLEMENTATION_GUIDE.md` - 如何实现新 kernel
   - `TEST_OPERATOR_USAGE.md` - 如何使用测试框架
3. **运行 Benchmark**: 对比不同量化方案的性能
4. **添加新算子**: 使用算子框架注册自己的实现
5. **贡献代码**: 提交 PR 改进项目

## 获取帮助

- 查看文档: `~/Agent4Kernel/KernelEvalPlus/README.md`
- 查看抽离报告: `~/Agent4Kernel/KernelEvalPlus/EXTRACTION_REPORT.md`
- 运行结构验证: `python3 verify_structure.py`
