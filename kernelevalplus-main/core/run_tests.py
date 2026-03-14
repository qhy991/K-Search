#!/usr/bin/env python3
"""
Unified test runner for KernelEvalPlus operator variants.

This script provides a single entry point for all testing operations,
replacing the fragmented test_*.py files with consistent interfaces.

Usage:
    # Run all tests
    python run_tests.py

    # Test specific variant
    python run_tests.py --variant w8a8c8_q8_0_fp32_int8

    # Test all DeepSeek-V3 variants
    python run_tests.py --model deepseek-v3

    # Pattern matching
    python run_tests.py --pattern "*q4_0*"

    # Include benchmarks
    python run_tests.py --benchmark

    # Save results to JSON
    python run_tests.py --output results.json

    # Custom test configuration
    python run_tests.py --variant w4a8_q4_0_fp32_int8 --M 128 --N 4096 --K 4096

    # Verbose output
    python run_tests.py --variant w8a8c8_q8_0_fp32_int8 --verbose

    # List available variants
    python run_tests.py --list
"""

import argparse
import fnmatch
import importlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Add python directory to path
PYTHON_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    variant: str
    passed: bool
    metric_name: str = "nmse"
    metric_value: float = float('inf')
    threshold: float = 0.1
    error: Optional[str] = None
    params: Optional[Dict] = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    variant: str
    M: int
    N: int
    K: int
    time_ms: float
    gflops: float


@dataclass
class VariantTestResults:
    """Results for a single variant."""
    variant: str
    description: str
    passed: int
    failed: int
    skipped: int
    tests: List[TestResult]
    benchmarks: Optional[List[BenchmarkResult]] = None


class TestRunner:
    """
    Unified test runner for operator variants.
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize the test runner.

        Args:
            project_root: Project root directory (auto-detected if None)
        """
        self.project_root = project_root or PROJECT_ROOT
        self.variants_dir = self.project_root / 'python' / 'operators' / 'quant_gemm' / 'variants'
        self._module = None
        self._torch = None

    def _ensure_torch(self):
        """Ensure torch is imported."""
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _get_module(self):
        """Get the compiled quant_gemm module."""
        if self._module is None:
            try:
                self._module = importlib.import_module('quant_gemm._C')
            except ImportError:
                print("Warning: quant_gemm._C not available. Run: pip install -e .")
                return None
        return self._module

    def find_variants(self, pattern: Optional[str] = None,
                      model: Optional[str] = None) -> List[Path]:
        """
        Find variant directories matching criteria.

        Args:
            pattern: Glob pattern for variant names
            model: Model architecture filter (e.g., 'deepseek-v3')

        Returns:
            List of variant directory paths
        """
        variants = []

        for spec_path in self.variants_dir.rglob('spec.json'):
            variant_dir = spec_path.parent

            # Load spec to get name and model info
            try:
                with open(spec_path) as f:
                    spec = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            name = spec.get('name', variant_dir.name)

            # Filter by pattern
            if pattern:
                if not (fnmatch.fnmatch(name, pattern) or
                        fnmatch.fnmatch(variant_dir.name, pattern)):
                    continue

            # Filter by model
            if model:
                model_archs = spec.get('model_architectures', [])
                if model.lower() not in [m.lower() for m in model_archs]:
                    # Also check directory path for model names
                    if model.lower().replace('-', '_') not in str(variant_dir).lower():
                        continue

            variants.append(variant_dir)

        return sorted(variants)

    def list_variants(self) -> List[Dict]:
        """
        List all available variants with their info.

        Returns:
            List of variant info dictionaries
        """
        variants = []

        for spec_path in self.variants_dir.rglob('spec.json'):
            variant_dir = spec_path.parent

            try:
                with open(spec_path) as f:
                    spec = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            variants.append({
                'name': spec.get('name', variant_dir.name),
                'description': spec.get('description', '')[:60],
                'family': spec.get('family', 'quant_gemm'),
                'path': str(variant_dir.relative_to(self.project_root)),
            })

        return sorted(variants, key=lambda x: x['name'])

    def load_spec(self, variant_dir: Path) -> dict:
        """Load spec.json from variant directory."""
        spec_path = variant_dir / 'spec.json'
        with open(spec_path) as f:
            return json.load(f)

    def get_quantizer(self, module, dtype: str):
        """Get quantizer function from module."""
        quantizer_map = {
            'block_q4_0': 'quantize_q4_0',
            'block_q4_1': 'quantize_q4_1',
            'block_q8_0': 'quantize_q8_0',
            'block_q8_1': 'quantize_q8_1',
            'block_q5_0': 'quantize_q5_0',
            'block_q5_1': 'quantize_q5_1',
        }

        func_name = quantizer_map.get(dtype)
        if func_name and hasattr(module, func_name):
            return getattr(module, func_name)
        return None

    def get_kernel(self, module, spec: dict):
        """Get kernel function from module."""
        kernel_info = spec.get('kernel', {})
        kernel_name = kernel_info.get('entry_point', spec['name'])

        # Try various naming patterns
        candidates = [
            kernel_name,
            f"gemm_{spec['name']}",
            spec['name'],
        ]

        for name in candidates:
            if hasattr(module, name):
                return getattr(module, name)

        return None

    def generate_inputs(self, spec: dict, params: dict, device: str = 'cuda') -> Dict:
        """Generate test inputs based on spec."""
        torch = self._ensure_torch()
        inputs = {}
        M, N, K = params['M'], params['N'], params['K']

        for name, tensor_info in spec.get('inputs', {}).items():
            dtype = tensor_info.get('dtype', 'float32')

            # Determine shape
            if 'weight' in name.lower():
                shape = (N, K)
            elif 'activation' in name.lower():
                shape = (M, K)
            else:
                # Parse shape from spec
                shape = self._resolve_shape(tensor_info.get('shape', []), params)

            # Generate tensor
            if dtype in ('float32',):
                inputs[name] = torch.randn(shape, dtype=torch.float32, device=device)
            elif dtype in ('float16',):
                inputs[name] = torch.randn(shape, dtype=torch.float16, device=device)
            elif dtype in ('bfloat16',):
                inputs[name] = torch.randn(shape, dtype=torch.bfloat16, device=device)
            else:
                # Quantized types - generate FP32 first
                inputs[name] = torch.randn(shape, dtype=torch.float32, device=device)

        return inputs

    def _resolve_shape(self, shape_spec: list, params: dict) -> tuple:
        """Resolve symbolic shape to concrete dimensions."""
        shape = []
        for dim in shape_spec:
            if isinstance(dim, int):
                shape.append(dim)
            elif dim in params:
                shape.append(params[dim])
            elif '/' in str(dim):
                parts = str(dim).split('/')
                val = params.get(parts[0], 4096)
                shape.append(val // int(parts[1]))
        return tuple(shape) if shape else (params.get('M', 1), params.get('K', 4096))

    def quantize_inputs(self, raw_inputs: dict, spec: dict, module) -> dict:
        """Quantize inputs according to spec."""
        quantized = {}

        for name, tensor in raw_inputs.items():
            tensor_info = spec.get('inputs', {}).get(name, {})
            dtype = tensor_info.get('dtype', 'float32')

            if dtype.startswith('block_q'):
                quantizer = self.get_quantizer(module, dtype)
                if quantizer is None:
                    raise RuntimeError(f"Quantizer not found for {dtype}")
                quantized[name] = quantizer(tensor)
            else:
                quantized[name] = tensor

        return quantized

    def compute_nmse(self, output, reference) -> float:
        """Compute Normalized Mean Squared Error."""
        torch = self._ensure_torch()

        output = output.float().flatten()
        reference = reference.float().flatten()

        mse = torch.mean((output - reference) ** 2).item()
        ref_var = torch.var(reference).item()

        if ref_var < 1e-10:
            return 0.0 if mse < 1e-10 else float('inf')

        return mse / ref_var

    def run_reference(self, raw_inputs: dict, quantized_inputs: dict,
                      params: dict, spec: dict, variant_dir: Path) -> 'torch.Tensor':
        """Run reference implementation."""
        torch = self._ensure_torch()

        # Try to load custom reference
        if 'reference' in spec:
            ref_spec = spec['reference']
            if ':' in ref_spec:
                ref_file, ref_func = ref_spec.split(':')
                ref_path = variant_dir / ref_file
                if ref_path.exists():
                    import importlib.util
                    spec_module = importlib.util.spec_from_file_location('reference', ref_path)
                    ref_module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(ref_module)

                    if hasattr(ref_module, ref_func):
                        func = getattr(ref_module, ref_func)
                        # Custom reference expects quantized inputs
                        return func(**quantized_inputs)

        # Default: FP32 matmul (uses raw inputs)
        weight = raw_inputs.get('weight')
        activation = raw_inputs.get('activation')

        if weight is not None and activation is not None:
            return torch.matmul(activation, weight.T)

        raise RuntimeError("No reference implementation available")

    def run_test(self, variant_dir: Path, config: dict,
                 device: str = 'cuda') -> TestResult:
        """Run a single test configuration."""
        torch = self._ensure_torch()
        module = self._get_module()

        if module is None:
            return TestResult(
                name=config.get('name', 'unknown'),
                variant=variant_dir.name,
                passed=False,
                error="quant_gemm._C module not available"
            )

        spec = self.load_spec(variant_dir)
        name = config.get('name', 'test')

        # Build params with defaults
        params = {k: v for k, v in config.items() if k not in ('name', 'description')}
        for param_name, param_info in spec.get('params', {}).items():
            if param_name not in params and 'default' in param_info:
                params[param_name] = param_info['default']

        try:
            # Generate and quantize inputs
            raw_inputs = self.generate_inputs(spec, params, device)
            quantized_inputs = self.quantize_inputs(raw_inputs, spec, module)

            # Run reference (pass both raw and quantized inputs)
            ref_output = self.run_reference(raw_inputs, quantized_inputs, params, spec, variant_dir)
            if ref_output.device.type != device:
                ref_output = ref_output.to(device)

            # Get kernel
            kernel = self.get_kernel(module, spec)
            if kernel is None:
                return TestResult(
                    name=name,
                    variant=spec.get('name', variant_dir.name),
                    passed=False,
                    error="Kernel not found"
                )

            # Run kernel
            M, N, K = params['M'], params['N'], params['K']
            weight_q = quantized_inputs['weight']
            activation_q = quantized_inputs['activation']

            # Check activation type for calling convention
            act_dtype = spec.get('inputs', {}).get('activation', {}).get('dtype', 'float32')
            is_quantized_act = act_dtype.startswith('block_q')

            if is_quantized_act:
                output = kernel(weight_q, activation_q, N, M, K)
                output = output.T
            else:
                output = kernel(weight_q, activation_q, M, N, K)

            # Compute accuracy
            accuracy_spec = spec.get('accuracy', {'metric': 'nmse', 'threshold': 0.1})
            threshold = accuracy_spec.get('threshold', 0.1)
            metric_value = self.compute_nmse(output, ref_output)
            passed = metric_value <= threshold

            return TestResult(
                name=name,
                variant=spec.get('name', variant_dir.name),
                passed=passed,
                metric_name='nmse',
                metric_value=metric_value,
                threshold=threshold,
                params=params
            )

        except Exception as e:
            return TestResult(
                name=name,
                variant=spec.get('name', variant_dir.name),
                passed=False,
                error=str(e)
            )

    def run_benchmark(self, variant_dir: Path, config: dict,
                      warmup: int = 10, iterations: int = 100,
                      device: str = 'cuda') -> Optional[BenchmarkResult]:
        """Run benchmark for a configuration."""
        torch = self._ensure_torch()
        module = self._get_module()

        if module is None:
            return None

        spec = self.load_spec(variant_dir)

        # Build params
        params = {k: v for k, v in config.items() if k not in ('name', 'description')}
        for param_name, param_info in spec.get('params', {}).items():
            if param_name not in params and 'default' in param_info:
                params[param_name] = param_info['default']

        M, N, K = params['M'], params['N'], params['K']

        try:
            # Generate inputs
            raw_inputs = self.generate_inputs(spec, params, device)
            quantized_inputs = self.quantize_inputs(raw_inputs, spec, module)

            kernel = self.get_kernel(module, spec)
            if kernel is None:
                return None

            weight_q = quantized_inputs['weight']
            activation_q = quantized_inputs['activation']

            act_dtype = spec.get('inputs', {}).get('activation', {}).get('dtype', 'float32')
            is_quantized_act = act_dtype.startswith('block_q')

            # Warmup
            for _ in range(warmup):
                if is_quantized_act:
                    _ = kernel(weight_q, activation_q, N, M, K)
                else:
                    _ = kernel(weight_q, activation_q, M, N, K)
            torch.cuda.synchronize()

            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iterations):
                if is_quantized_act:
                    _ = kernel(weight_q, activation_q, N, M, K)
                else:
                    _ = kernel(weight_q, activation_q, M, N, K)
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end) / iterations
            flops = 2 * M * N * K
            gflops = flops / (elapsed_ms * 1e-3) / 1e9

            return BenchmarkResult(
                name=config.get('name', 'bench'),
                variant=spec.get('name', variant_dir.name),
                M=M, N=N, K=K,
                time_ms=elapsed_ms,
                gflops=gflops
            )

        except Exception as e:
            print(f"Benchmark error: {e}")
            return None

    def run_variant(self, variant_dir: Path,
                    custom_params: Optional[Dict] = None,
                    benchmark: bool = False,
                    verbose: bool = False) -> VariantTestResults:
        """
        Run all tests for a variant.

        Args:
            variant_dir: Path to variant directory
            custom_params: Override test parameters (M, N, K)
            benchmark: Include benchmarks
            verbose: Print detailed output

        Returns:
            VariantTestResults with all test results
        """
        spec = self.load_spec(variant_dir)
        variant_name = spec.get('name', variant_dir.name)

        if verbose:
            print(f"\nTesting: {variant_name}")
            print(f"  {spec.get('description', '')[:60]}")

        # Get test configs
        if custom_params:
            configs = [{'name': 'custom', **custom_params}]
        else:
            configs = spec.get('test_configs', [
                {'name': 'default', 'M': 1, 'N': 4096, 'K': 4096}
            ])

        # Run tests
        tests = []
        passed = 0
        failed = 0
        skipped = 0

        for config in configs:
            result = self.run_test(variant_dir, config)
            tests.append(result)

            if result.passed:
                passed += 1
                if verbose:
                    print(f"  [PASS] {result.name}: nmse={result.metric_value:.2e}")
            else:
                failed += 1
                if verbose:
                    if result.error:
                        print(f"  [FAIL] {result.name}: {result.error}")
                    else:
                        print(f"  [FAIL] {result.name}: nmse={result.metric_value:.2e} > {result.threshold}")

        # Run benchmarks if requested
        benchmarks = None
        if benchmark:
            benchmarks = []
            for config in configs:
                bench_result = self.run_benchmark(variant_dir, config)
                if bench_result:
                    benchmarks.append(bench_result)
                    if verbose:
                        print(f"  [BENCH] {bench_result.name}: {bench_result.time_ms:.3f}ms, {bench_result.gflops:.1f} GFLOPS")

        return VariantTestResults(
            variant=variant_name,
            description=spec.get('description', ''),
            passed=passed,
            failed=failed,
            skipped=skipped,
            tests=tests,
            benchmarks=benchmarks
        )

    def run_all(self, pattern: Optional[str] = None,
                model: Optional[str] = None,
                custom_params: Optional[Dict] = None,
                benchmark: bool = False,
                verbose: bool = False) -> List[VariantTestResults]:
        """
        Run tests for all matching variants.

        Args:
            pattern: Variant name pattern filter
            model: Model architecture filter
            custom_params: Override test parameters
            benchmark: Include benchmarks
            verbose: Print detailed output

        Returns:
            List of VariantTestResults
        """
        variants = self.find_variants(pattern, model)
        results = []

        for variant_dir in variants:
            result = self.run_variant(
                variant_dir,
                custom_params=custom_params,
                benchmark=benchmark,
                verbose=verbose
            )
            results.append(result)

        return results


def print_summary(results: List[VariantTestResults]):
    """Print test results summary."""
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for result in results:
        status = "PASS" if result.failed == 0 else "FAIL"
        print(f"[{status}] {result.variant}: {result.passed} passed, {result.failed} failed")
        total_passed += result.passed
        total_failed += result.failed
        total_skipped += result.skipped

    print("-" * 70)
    print(f"Total: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print("=" * 70)


def save_results(results: List[VariantTestResults], output_path: Path):
    """Save results to JSON file."""
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [
            {
                'variant': r.variant,
                'description': r.description,
                'passed': r.passed,
                'failed': r.failed,
                'skipped': r.skipped,
                'tests': [asdict(t) for t in r.tests],
                'benchmarks': [asdict(b) for b in r.benchmarks] if r.benchmarks else None
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified test runner for KernelEvalPlus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_tests.py                              # Run all tests
  python run_tests.py --variant w8a8c8_q8_0_fp32_int8   # Test specific variant
  python run_tests.py --model deepseek-v3          # Test model variants
  python run_tests.py --pattern "*q4_0*"           # Pattern matching
  python run_tests.py --benchmark                  # Include benchmarks
  python run_tests.py --output results.json        # Save results
  python run_tests.py --list                       # List variants
'''
    )

    # Test selection
    parser.add_argument('--variant', '-v', type=str,
                        help='Test specific variant by name')
    parser.add_argument('--pattern', '-p', type=str,
                        help='Filter variants by glob pattern')
    parser.add_argument('--model', '-m', type=str,
                        help='Filter by model architecture (e.g., deepseek-v3)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available variants')

    # Test options
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Include performance benchmarks')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # Custom parameters
    parser.add_argument('--M', type=int, help='Batch dimension')
    parser.add_argument('--N', type=int, help='Output features')
    parser.add_argument('--K', type=int, help='Input features')

    # Output
    parser.add_argument('--output', '-o', type=str,
                        help='Save results to JSON file')

    args = parser.parse_args()

    runner = TestRunner()

    # Handle --list
    if args.list:
        variants = runner.list_variants()
        print(f"Available variants ({len(variants)}):\n")

        for v in variants:
            print(f"  {v['name']}")
            if v['description']:
                print(f"    {v['description']}")

        return 0

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Tests may fail.")
    except ImportError:
        print("Error: PyTorch not found. Install with: pip install torch")
        return 1

    # Build custom params
    custom_params = None
    if args.M or args.N or args.K:
        custom_params = {}
        if args.M:
            custom_params['M'] = args.M
        if args.N:
            custom_params['N'] = args.N
        if args.K:
            custom_params['K'] = args.K

    # Handle --variant
    if args.variant:
        variants = runner.find_variants(pattern=args.variant)
        if not variants:
            print(f"Error: Variant '{args.variant}' not found")
            print("\nAvailable variants:")
            for v in runner.list_variants()[:10]:
                print(f"  - {v['name']}")
            return 1

        variant_dir = variants[0]
        result = runner.run_variant(
            variant_dir,
            custom_params=custom_params,
            benchmark=args.benchmark,
            verbose=True
        )
        results = [result]
    else:
        # Run all matching
        results = runner.run_all(
            pattern=args.pattern,
            model=args.model,
            custom_params=custom_params,
            benchmark=args.benchmark,
            verbose=args.verbose
        )

    if not results:
        print("No variants found matching criteria.")
        return 1

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        save_results(results, Path(args.output))

    # Return exit code based on results
    total_failed = sum(r.failed for r in results)
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
