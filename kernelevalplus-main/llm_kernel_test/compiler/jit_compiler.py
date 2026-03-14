#!/usr/bin/env python3
"""
JIT 编译器模块

提供 CUDA kernel 的 JIT 编译功能，参考 robust-kbench 实现。
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List


def get_cuda_gencode_flags():
    """获取当前 GPU 的 CUDA gencode 标志（参考 robust-kbench）

    使用 gencode 格式而不是 -arch，因为：
    - PyTorch 对 -arch 的处理可能有问题
    - gencode 格式对 sm_120 (RTX 50 系列) 也能正常工作

    Returns:
        gencode 标志列表，格式: ['-gencode=arch=compute_XX,code=sm_XX', ...]
    """
    try:
        import torch
        device_props = torch.cuda.get_device_properties(0)
        major = device_props.major
        minor = device_props.minor

        # 使用 gencode 格式：-gencode=arch=compute_XX,code=sm_XX
        gencode_flags = [
            f'-gencode=arch=compute_{major}{minor},code=compute_{major}{minor}',
            f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}',
        ]

        print(f"  🎯 检测到 GPU: {device_props.name}, Compute: {major}.{minor}")
        print(f"  📋 使用 gencode: compute_{major}{minor}, sm_{major}{minor}")
        return gencode_flags
    except Exception as e:
        print(f"  ⚠️  无法检测 GPU 架构: {e}, 使用默认 sm_89")
        return [
            '-gencode=arch=compute_89,code=compute_89',
            '-gencode=arch=compute_89,code=sm_89',
        ]


class JITCompiler:
    """CUDA kernel JIT 编译器"""

    def __init__(self, compiled_modules: Optional[Dict] = None):
        """
        Args:
            compiled_modules: 已编译模块的缓存字典
        """
        if compiled_modules is None:
            compiled_modules = {}
        self._compiled_modules = compiled_modules

    def compile_kernel(
        self,
        attempt_dir: Path,
        spec: Optional[Dict] = None,
        name: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        使用 JIT 编译 CUDA 代码 (参考 robust-kbench)

        Args:
            attempt_dir: 包含 kernel.cu 的目录
            spec: 可选的 spec 字典（用于获取 kernel 入口点信息）
            name: 可选的模块名称
            verbose: 是否打印详细输出

        Returns:
            编译结果字典，包含:
            - success: bool
            - errors: list of str
            - warnings: list of str
            - diagnostics: list of dict (如果编译失败)
        """
        result = {
            "success": False,
            "errors": [],
            "warnings": []
        }

        kernel_file = attempt_dir / "kernel.cu"
        if not kernel_file.exists():
            result["errors"].append("kernel.cu not found")
            return result

        bindings_file = attempt_dir / "bindings.cpp"
        if not bindings_file.exists():
            result["warnings"].append("bindings.cpp not found, creating default")

        try:
            import torch
            from torch.utils.cpp_extension import load
        except ImportError as e:
            result["errors"].append(
                f"PyTorch not available. Please install:\n"
                f"  pip install torch>=2.0.0\n"
                f"Or use project requirements:\n"
                f"  pip install -r python/requirements.txt"
            )
            return result

        # 获取 CUDA gencode 标志
        gencode_flags = get_cuda_gencode_flags()

        # 构建编译标志 (参考 robust-kbench 的 cuda_compile.py)
        nvcc_flags = [
            '-O3',
            '--use_fast_math',
        ] + gencode_flags  # 添加 gencode 标志

        try:
            # 获取模块名称
            if name is None:
                if spec:
                    name = spec.get('name', attempt_dir.parent.name)
                else:
                    name = attempt_dir.parent.name
            else:
                name = spec.get('name', name) if spec else name

            # 准备源文件列表
            sources = [str(kernel_file)]

            # 读取并处理 kernel 代码
            kernel_code, fixes_applied = self._preprocess_kernel(kernel_file, spec)

            # 检查是否需要生成 PyTorch wrapper
            if not kernel_code:
                kernel_code = self._read_kernel_code(kernel_file)

            has_pybind = 'PYBIND11_MODULE' in kernel_code

            if not has_pybind:
                wrapper_file = self._generate_pytouch_wrapper(
                    attempt_dir, kernel_code, spec
                )
                if wrapper_file:
                    sources.append(str(wrapper_file))

            # 使用 JIT 编译 (参考 robust-kbench 方式)
            if verbose:
                print(f"  🔨 JIT 编译中...")

            module = load(
                name=f"{name}_test",
                sources=sources,
                extra_cflags=['-O3'],
                extra_cuda_cflags=nvcc_flags,
                verbose=verbose,
            )

            # 保存编译后的模块引用供后续测试使用
            self._compiled_modules[str(attempt_dir)] = module

            result["success"] = True
            result["warnings"].append("JIT compilation successful")

        except ImportError as e:
            result["errors"].append(f"PyTorch not available: {e}")
        except Exception as e:
            error_msg = str(e)
            result["errors"].append(error_msg)

        return result

    def _read_kernel_code(self, kernel_file: Path) -> str:
        """读取 kernel 源代码"""
        with open(kernel_file, 'r') as f:
            return f.read()

    def _preprocess_kernel(self, kernel_file: Path, spec: Optional[Dict]) -> tuple:
        """
        预处理 kernel 代码：验证和修复 block size

        Returns:
            (kernel_code, fixes_applied)
        """
        with open(kernel_file, 'r') as f:
            kernel_code = f.read()

        fixes_applied = []

        # 验证 block size 是否有效（防止 CUDA 启动失败）
        block_size_fix_needed = False
        block_size_fixes = []

        # 查找所有 dim3 block(...) 声明
        block_matches = re.finditer(r'dim3\s+block\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', kernel_code)
        for match in block_matches:
            x = int(match.group(1))
            y = int(match.group(2))
            total_threads = x * y
            if total_threads > 1024:
                block_size_fix_needed = True
                # 建议修复
                if x >= y:
                    new_x = min(x, 1024)
                    new_y = 1024 // new_x
                else:
                    new_y = min(y, 1024)
                    new_x = 1024 // new_y
                block_size_fixes.append(
                    f"  ❌ dim3 block({x}, {y}) = {total_threads} threads (exceeds 1024 limit)"
                )
                block_size_fixes.append(
                    f"  ✅ Should be: dim3 block({new_x}, {new_y}) = {new_x * new_y} threads"
                )

        if block_size_fix_needed:
            print(f"  ⚠️  检测到无效的 block size 配置:")
            for fix in block_size_fixes:
                print(fix)
            print(f"  🔧 自动修复中...")
            # 替换所有无效的 block size
            def fix_block_size(match):
                x = int(match.group(1))
                y = int(match.group(2))
                total = x * y
                if total > 1024:
                    # 计算安全的配置
                    if x >= y:
                        new_x = min(x, 1024)
                        new_y = max(1, 1024 // new_x)
                    else:
                        new_y = min(y, 1024)
                        new_x = max(1, 1024 // new_y)
                    return f'dim3 block({new_x}, {new_y})'
                return match.group(0)

            kernel_code = re.sub(
                r'dim3\s+block\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
                fix_block_size,
                kernel_code
            )
            # 写回修复后的代码
            with open(kernel_file, 'w') as f:
                f.write(kernel_code)
            print(f"  ✅ 已自动修复 block size")
            fixes_applied.append("block_size_fixed")

        return kernel_code, fixes_applied

    def _generate_pytouch_wrapper(
        self,
        attempt_dir: Path,
        kernel_code: str,
        spec: Optional[Dict]
    ) -> Optional[Path]:
        """
        为 extern "C" 风格的 kernel 生成 PyTorch wrapper

        Returns:
            wrapper_file 路径，如果不需要生成则返回 None
        """
        if 'PYBIND11_MODULE' in kernel_code:
            return None

        print(f"  ⚠️  kernel.cu 没有 PYBIND11_MODULE，自动生成 wrapper...")

        # 从 spec 读取入口点名称和激活类型
        entry_point_name = "forward"
        activation_is_fp32 = True

        if spec:
            kernel_info = spec.get("kernel", {})
            entry_point_name = kernel_info.get("entry_point", "forward")
            act_dtype = spec.get("inputs", {}).get("activation", {}).get("dtype", "float32")
            activation_is_fp32 = act_dtype in ("float32", "float16", "bfloat16", "fp32")

        act_cpp_type = "float" if activation_is_fp32 else "uint8_t"

        # 从 kernel.cu 中提取 extern "C" 函数签名
        # 支持常见模式：extern "C" void func_name(...)
        extern_c_match = re.search(
            r'extern\s+"C"\s+void\s+(\w+)\s*\(',
            kernel_code
        )
        if extern_c_match:
            entry_point_name = extern_c_match.group(1)
            print(f"  📍 检测到 extern \"C\" 函数: {entry_point_name}")

        # Use format() instead of f-string to avoid escaping issues
        wrapper_code = '''#include <torch/extension.h>
#include <cstdint>

// Forward declaration of the extern "C" kernel wrapper
extern "C" void {entry_point_name}(
    const uint8_t* weight,
    const {act_cpp_type}* activation,
    float* output,
    int M, int N, int K
);

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({{M, N}}, torch::dtype(torch::kFloat32).device(weight.device()));

    {entry_point_name}(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<{act_cpp_type}>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &forward, "Auto-wrapped kernel");
    m.def("{entry_point_name}", &forward, "Auto-wrapped kernel (entry_point alias)");
}}
'''.format(entry_point_name=entry_point_name, act_cpp_type=act_cpp_type)
        wrapper_file = attempt_dir / "auto_wrapper.cu"
        with open(wrapper_file, "w") as f:
            f.write(wrapper_code)
        print(f"  ✅ 自动生成 wrapper: auto_wrapper.cu")

        return wrapper_file

    def get_compiled_module(self, attempt_dir: Path) -> Optional[Any]:
        """获取已编译的模块"""
        return self._compiled_modules.get(str(attempt_dir))

    def has_compiled_module(self, attempt_dir: Path) -> bool:
        """检查是否已编译指定目录的模块"""
        return str(attempt_dir) in self._compiled_modules
