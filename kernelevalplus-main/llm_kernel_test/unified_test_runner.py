#!/usr/bin/env python3
"""
统一测试运行器 - 支持多种算子类型

支持的算子:
- Quantized GEMM (q4_0, q4_1, q8_0, q8_1)
- Flash Attention
- RMS Norm
- TopK

使用方法:
    # 统一测试命令格式
    python llm_kernel_test/unified_test_runner.py --test \\
        --definition definitions/<op_type>/<model>/<name>.json \\
        --attempt-path attempts/<name>_v<N>

    # Flash Attention 测试
    python llm_kernel_test/unified_test_runner.py --test \\
        --definition definitions/flash_attention/llama/fp32_flash_attention_llama3_8b_f16_cache512.json \\
        --attempt-path attempts/flash_attn_v1

    # Quantized GEMM 测试
    python llm_kernel_test/unified_test_runner.py --test \\
        --definition definitions/quant_gemm/llama/w4a32c8_q4_0_llama3_8b.json \\
        --attempt-path attempts/gemm_v1

    # RMS Norm 测试
    python llm_kernel_test/unified_test_runner.py --test \\
        --definition definitions/rms_norm/llama/fp32_rms_norm_llama3_8b_hs4096.json \\
        --attempt-path attempts/rms_norm_v1
"""
import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Import handler framework
from llm_kernel_test.op_test_handler import (
    get_handler, detect_op_type,
    FlashAttentionHandler, RMSNormHandler, TopKHandler, QuantGEMMHandler
)

# Import compiler utilities
from llm_kernel_test.compiler import JITCompiler, ErrorAnalyzer, get_cuda_gencode_flags


class UnifiedTestRunner:
    """统一测试运行器"""

    def __init__(self, config_path="llm_kernel_test/test_config.json"):
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.test_root = self.project_root / "llm_kernel_test"
        self.definition_path = None

        # Initialize components
        self.compiler = JITCompiler()
        self.error_analyzer = ErrorAnalyzer()
        self._compiled_modules = {}
        self._spec_cache = {}

    def _load_config(self, config_path):
        """加载配置"""
        default_config = {
            "sandbox_dir": "llm_kernel_test/sandbox",
            "benchmark_config": {"warmup": 10, "iterations": 100},
            "correctness_threshold": {"nmse": 0.1}
        }

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                default_config.update(config)

        return default_config

    def _load_spec(self, attempt_dir: Path) -> Dict:
        """加载 spec（优先 spec.json，其次 definition JSON）"""
        cache_key = str(attempt_dir)
        if cache_key in self._spec_cache:
            return self._spec_cache[cache_key]

        spec = None

        # 1. 尝试 spec.json
        spec_file = attempt_dir / "spec.json"
        if spec_file.exists():
            with open(spec_file) as f:
                spec = json.load(f)

        # 2. 使用外部 definition JSON
        if spec is None and self.definition_path:
            with open(self.definition_path) as f:
                definition = json.load(f)
            spec = self._convert_definition_to_spec(definition)

        if spec:
            self._spec_cache[cache_key] = spec
            return spec

        raise FileNotFoundError(f"No spec found in {attempt_dir}")

    def _convert_definition_to_spec(self, definition: Dict) -> Dict:
        """转换 definition 格式到内部 spec 格式"""
        # 简单转换，保持原始 definition 信息
        spec = dict(definition)

        # 确保有 params 字段（从 axes 转换）
        if "params" not in spec and "axes" in spec:
            spec["params"] = {}
            for name, info in spec["axes"].items():
                if isinstance(info, dict):
                    if info.get("type") == "const":
                        spec["params"][name] = {"value": info.get("value")}
                    else:
                        spec["params"][name] = {"default": 1}

        return spec

    def test(self, attempt_id: str, variant: str, attempt_path: str = None,
            definition_path: str = None):
        """运行测试"""
        if definition_path:
            self.definition_path = definition_path

        print(f"🧪 运行测试: {attempt_id}")

        # 确定测试目录
        if attempt_path:
            attempt_dir = Path(attempt_path)
            if not (attempt_dir / "kernel.cu").exists():
                attempt_dir = attempt_dir / variant
        else:
            attempt_dir = self.project_root / self.config["sandbox_dir"] / "generated" / attempt_id / variant

        if not attempt_dir.exists():
            print(f"❌ 找不到测试目录: {attempt_dir}")
            return {"error": "Test directory not found"}

        # 加载 spec
        spec = self._load_spec(attempt_dir)

        # 检测算子类型
        op_type = detect_op_type(spec)
        handler = get_handler(op_type)

        print(f"  📋 算子类型: {op_type}")

        results = {
            "attempt_id": attempt_id,
            "variant": variant,
            "op_type": op_type,
            "tested_at": datetime.now().isoformat(),
            "spec": spec,
            "compilation": {},
            "correctness": {},
            "performance": {}
        }

        # 1. 编译
        print("\n📦 步骤 1: 编译检查")
        compile_result = self._compile(attempt_dir, spec, op_type)
        results["compilation"] = compile_result

        if not compile_result["success"]:
            print("❌ 编译失败")
            self._save_results(attempt_dir, results)
            return results

        print("✅ 编译成功")

        # 2. 正确性测试
        print("\n✅ 步骤 2: 正确性测试")
        correctness_result = self._test_correctness(attempt_dir, spec, handler, op_type)
        results["correctness"] = correctness_result

        if not correctness_result["passed"]:
            print("❌ 正确性测试失败")
            self._save_results(attempt_dir, results)
            return results

        print("✅ 正确性测试通过")

        # 3. 性能测试
        print("\n🚀 步骤 3: 性能测试")
        performance_result = self._test_performance(attempt_dir, spec, handler, op_type)
        results["performance"] = performance_result

        print("✅ 性能测试完成")

        # 保存结果
        self._save_results(attempt_dir, results)
        self._print_summary(results)

        return results

    def _compile(self, attempt_dir: Path, spec: Dict, op_type: str) -> Dict:
        """编译 kernel - 使用 Handler 高级功能"""
        result = {"success": False, "errors": [], "warnings": [], "diagnostics": []}

        kernel_file = attempt_dir / "kernel.cu"
        if not kernel_file.exists():
            result["errors"].append("kernel.cu not found")
            return result

        # 获取 handler
        handler = get_handler(op_type)
        if not handler:
            result["errors"].append(f"Unknown operator type: {op_type}")
            return result

        try:
            from torch.utils.cpp_extension import load

            # 读取 kernel 代码
            with open(kernel_file) as f:
                kernel_code = f.read()

            # 1. 预处理 kernel（如 block size 修复）
            processed_code = handler.preprocess_kernel(kernel_code)
            if processed_code != kernel_code:
                with open(kernel_file, 'w') as f:
                    f.write(processed_code)
                kernel_code = processed_code
                print("  ✅ 已预处理 kernel 代码")

            # 2. 获取 GPU 架构
            gencode_flags = get_cuda_gencode_flags()
            nvcc_flags = ['-O3', '--use_fast_math'] + gencode_flags

            name = spec.get('name', attempt_dir.parent.name)
            # Sanitize name for C++ identifier
            name = name.replace('-', '_').replace('.', '_')  # Replace hyphens and dots
            # Remove common suffixes
            if name.endswith('_json'):
                name = name[:-5]  # Remove _json suffix
            sources = [str(kernel_file)]

            # 3. 检查是否需要生成 wrapper（使用 handler 的方法）
            has_pybind = 'PYBIND11_MODULE' in kernel_code

            if not has_pybind:
                wrapper_code = handler.generate_wrapper(spec, kernel_code)
                if not wrapper_code:
                    # Fallback to default wrapper generation
                    wrapper_code = self._generate_wrapper(spec, op_type)

                if wrapper_code:
                    wrapper_file = attempt_dir / "auto_wrapper.cu"
                    with open(wrapper_file, 'w') as f:
                        f.write(wrapper_code)
                    sources.append(str(wrapper_file))
                    print(f"  ✅ 生成 {op_type} wrapper")

            # 4. JIT 编译
            print("  🔨 JIT 编译中...")
            module = load(
                name=f"{name}_{op_type}_test",
                sources=sources,
                extra_cflags=['-O3'],
                extra_cuda_cflags=nvcc_flags,
                verbose=True
            )

            self._compiled_modules[str(attempt_dir)] = module
            result["success"] = True

        except Exception as e:
            error_msg = str(e)
            result["errors"].append(error_msg)
            print(f"  ❌ 编译错误: {e}")

            # 5. 使用 handler 诊断错误
            diagnostics = handler.diagnose_compilation_error(error_msg, kernel_code)
            if diagnostics:
                result["diagnostics"] = diagnostics
                print(f"\n  🔍 编译错误诊断:")
                for diag in diagnostics:
                    print(f"     问题: {diag['problem']}")
                    print(f"     建议: {diag['suggestion']}")

        return result

    def _generate_wrapper(self, spec: Dict, op_type: str) -> Optional[str]:
        """根据算子类型生成 wrapper"""
        if op_type == "flash_attention":
            return '''#include <torch/extension.h>

extern "C" void flash_attn_kernel(
    const float* query,
    const void* key_cache,
    const void* value_cache,
    float* output,
    int batch_size, int seq_len, int num_heads, int head_dim
);

torch::Tensor forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    auto output = torch::empty({batch_size, num_heads, head_dim},
        torch::dtype(torch::kFloat32).device(query.device()));

    flash_attn_kernel(
        query.data_ptr<float>(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        output.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention");
}
'''
        elif op_type == "rms_norm":
            return '''#include <torch/extension.h>

extern "C" void rms_norm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size, int hidden_size
);

torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor weight) {
    int batch_size = hidden_states.size(0);
    int hidden_size = hidden_states.size(1);

    auto output = torch::empty({batch_size, hidden_size},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    rms_norm_kernel(
        hidden_states.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, hidden_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RMS Norm");
}
'''
        elif op_type == "topk":
            return '''#include <torch/extension.h>

extern "C" void topk_kernel(
    const float* probs,
    int64_t* indices,
    int batch_size, int vocab_size, int k
);

torch::Tensor forward(torch::Tensor probs, int k) {
    int batch_size = probs.size(0);
    int vocab_size = probs.size(1);

    auto indices = torch::empty({batch_size, k},
        torch::dtype(torch::kInt64).device(probs.device()));

    topk_kernel(
        probs.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch_size, vocab_size, k
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "TopK");
}
'''
        return None

    def _test_correctness(self, attempt_dir: Path, spec: Dict, handler, op_type: str) -> Dict:
        """正确性测试 - 使用 Handler 的参考实现"""
        result = {"passed": False, "test_cases": []}

        try:
            module = self._compiled_modules[str(attempt_dir)]
            kernel_func = getattr(module, "forward", None)

            if not kernel_func:
                result["errors"] = ["forward function not found"]
                return result

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 使用 handler 获取测试配置
            test_configs = handler.get_test_configs(spec)
            if not test_configs:
                test_configs = spec.get("test_configs", [{"name": "smoke_test"}])

            threshold = spec.get("accuracy", {}).get("threshold", 0.05)

            all_passed = True
            all_nmse = []

            for test_config in test_configs:
                try:
                    # 使用 handler 生成输入
                    inputs = handler.generate_inputs(spec, test_config, device)

                    # 运行 kernel
                    output = handler.run_kernel(kernel_func, inputs, spec)

                    # 使用 handler 计算参考输出
                    ref_output = handler.get_reference_output(spec, inputs)

                    # 使用 handler 验证输出形状
                    output_cpu = output.cpu().float()
                    ref_cpu = ref_output.cpu().float()
                    output_cpu = handler.validate_output_shape(output_cpu, ref_cpu)

                    # 计算 NMSE
                    mse = torch.mean((output_cpu - ref_cpu) ** 2).item()
                    ref_power = torch.mean(ref_cpu ** 2).item()
                    nmse = mse / (ref_power + 1e-10)

                    # 检查通过条件
                    passed = nmse <= threshold and not torch.isnan(output).any() and not torch.isinf(output).any()
                    if not passed:
                        all_passed = False

                    all_nmse.append(nmse)

                    test_name = test_config.get("name", str(test_config))
                    result["test_cases"].append({
                        "config": test_name,
                        "passed": passed,
                        "nmse": round(nmse, 6)
                    })

                    status = "✅" if passed else "❌"
                    print(f"  📊 {test_name}: {status} NMSE={nmse:.6f} (threshold={threshold})")

                except Exception as e:
                    all_passed = False
                    result["test_cases"].append({
                        "config": test_config.get("name", str(test_config)),
                        "passed": False,
                        "error": str(e)
                    })
                    print(f"  ❌ {test_config.get('name', str(test_config))}: {e}")

            result["passed"] = all_passed
            if all_nmse:
                result["nmse"] = round(sum(all_nmse) / len(all_nmse), 6)

        except Exception as e:
            result["errors"] = [str(e)]
            import traceback
            traceback.print_exc()

        return result

    def _test_performance(self, attempt_dir: Path, spec: Dict, handler, op_type: str) -> Dict:
        """性能测试 - 使用 Handler 的性能计算"""
        result = {"benchmarks": [], "baseline_comparison": None}

        try:
            module = self._compiled_modules[str(attempt_dir)]
            kernel_func = getattr(module, "forward", None)

            if not kernel_func or not torch.cuda.is_available():
                result["errors"] = ["Kernel or CUDA not available"]
                return result

            device = "cuda"
            benchmark_config = self.config.get("benchmark_config", {})
            warmup = benchmark_config.get("warmup", 10)
            iterations = benchmark_config.get("iterations", 100)

            # 使用 handler 获取测试配置
            test_configs = handler.get_test_configs(spec)
            if not test_configs:
                test_configs = spec.get("test_configs", [])

            first_benchmark = None

            for test_config in test_configs:
                try:
                    # 使用 handler 生成输入
                    inputs = handler.generate_inputs(spec, test_config, device)

                    # Warmup
                    for _ in range(warmup):
                        _ = handler.run_kernel(kernel_func, inputs, spec)
                    torch.cuda.synchronize()

                    # Benchmark
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    for _ in range(iterations):
                        _ = handler.run_kernel(kernel_func, inputs, spec)
                    end_event.record()
                    end_event.synchronize()

                    avg_time_ms = start_event.elapsed_time(end_event) / iterations

                    # 使用 handler 计算性能
                    perf = handler.calculate_performance(None, avg_time_ms, spec, test_config)

                    test_name = test_config.get("name", str(test_config))
                    benchmark_result = {
                        "config": test_name,
                        "latency_ms": round(avg_time_ms, 3),
                        **perf
                    }

                    result["benchmarks"].append(benchmark_result)

                    if first_benchmark is None:
                        first_benchmark = {"perf": perf, "config": test_config}

                    perf_str = ", ".join(f"{k}={v}" for k, v in perf.items())
                    print(f"  ⚡ {test_name}: {avg_time_ms:.3f} ms, {perf_str}")

                except Exception as e:
                    result["benchmarks"].append({
                        "config": test_config.get("name", str(test_config)),
                        "error": str(e)
                    })
                    print(f"  ❌ {test_config.get('name', str(test_config))}: {e}")

            # Baseline 对比
            if first_benchmark:
                gpu_info = self._get_gpu_info()
                baseline = handler.query_baseline(spec, gpu_info["baseline_hardware"], first_benchmark["config"])

                if baseline:
                    perf = first_benchmark["perf"]
                    metric = handler.performance_metric
                    current_value = perf.get(metric, 0)
                    baseline_value = baseline.get(metric, 0)

                    if baseline_value and baseline_value > 0:
                        ratio = current_value / baseline_value * 100

                        result["baseline_comparison"] = {
                            "current_hardware": gpu_info["name"],
                            "baseline_hardware": gpu_info["baseline_hardware"],
                            f"current_{metric}": current_value,
                            f"baseline_{metric}": baseline_value,
                            "performance_ratio": round(ratio, 1),
                            "better": ratio >= 100
                        }

                        print(f"\n  📈 Baseline 对比:")
                        print(f"     当前: {current_value:.3f} {metric.upper()}")
                        print(f"     基线: {baseline_value:.3f} {metric.upper()}")
                        print(f"     比率: {ratio:.1f}% ", end="")
                        if ratio >= 100:
                            print("✅ 超越基线!")
                        elif ratio >= 80:
                            print("🟡 接近基线")
                        else:
                            print("🔴 低于基线")

        except Exception as e:
            result["errors"] = [str(e)]
            import traceback
            traceback.print_exc()

        return result

    def _get_gpu_info(self) -> Dict:
        """获取 GPU 信息"""
        try:
            props = torch.cuda.get_device_properties(0)
            name = props.name
            key = name.replace(" ", "").replace("NVIDIA", "").replace("Laptop", "").replace("Mobile", "")

            baseline_map = {
                "RTX4090": "RTX4090",
                "RTX4070": "RTX4070",
                "A100": "A100",
                "RTX5090": "RTX5090",
            }

            baseline_hardware = "RTX4090"  # default
            for model, hw in baseline_map.items():
                if model in key:
                    baseline_hardware = hw
                    break

            return {
                "name": name,
                "key": key,
                "baseline_hardware": baseline_hardware
            }
        except:
            return {"name": "Unknown", "baseline_hardware": "RTX4090"}

    def _save_results(self, attempt_dir: Path, results: Dict):
        """保存结果"""
        results_file = attempt_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        # 处理相对路径问题
        try:
            rel_path = results_file.relative_to(self.project_root)
            print(f"\n💾 结果已保存: {rel_path}")
        except ValueError:
            print(f"\n💾 结果已保存: {results_file.absolute()}")

    def _print_summary(self, results: Dict):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("📊 测试摘要")
        print("=" * 60)
        print(f"\n算子类型: {results['op_type']}")
        print(f"编译: {'✅ 成功' if results['compilation']['success'] else '❌ 失败'}")

        if results['compilation']['success']:
            print(f"正确性: {'✅ 通过' if results['correctness']['passed'] else '❌ 失败'}")

            if results['correctness']['passed']:
                print(f"性能:")
                for bench in results['performance']['benchmarks']:
                    perf_str = ", ".join(f"{k}={v}" for k, v in bench.items() if k != "config")
                    print(f"  {bench['config']}: {perf_str}")

                if results['performance'].get('baseline_comparison'):
                    bc = results['performance']['baseline_comparison']
                    print(f"\n📈 Baseline 对比: {bc['performance_ratio']}%")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="统一测试运行器")

    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--variant", type=str, help="变体名称")
    parser.add_argument("--attempt-id", type=str, help="尝试 ID")
    parser.add_argument("--attempt-path", type=str, help="尝试目录路径")
    parser.add_argument("--definition", type=str, help="Definition JSON 路径")

    args = parser.parse_args()

    runner = UnifiedTestRunner()

    if args.test:
        if not args.attempt_id and not args.attempt_path:
            print("❌ 需要指定 --attempt-id 或 --attempt-path")
            sys.exit(1)

        runner.test(
            args.attempt_id or "test",
            args.variant or "default",
            args.attempt_path,
            args.definition
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
