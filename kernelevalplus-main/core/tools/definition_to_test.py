#!/usr/bin/env python3
"""
Definition to Test Pipeline

从 definition 文件自动创建测试目录、生成 prompt 和运行测试。

Usage:
    # 为单个 definition 创建测试
    python -m python.tools.definition_to_test --definition definitions/quant_gemm/llama/w8a32c8_q8_0_fp32_int8_llama3_8b_att_qkv_n12288_k4096.json

    # 为所有 definitions 创建测试
    python -m python.tools.definition_to_test --all

    # 创建测试并运行
    python -m python.tools.definition_to_test --all --test

    # 使用特定模型作为 kernel 模板
    python -m python.tools.definition_to_test --all --kernel-template deepseek_v3
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.tools.prompt_generator import PromptGenerator


class DefinitionToTestPipeline:
    """从 Definition 到测试的自动化流程"""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)
        self.definitions_dir = self.project_root / "definitions"
        self.sandbox_dir = self.project_root / "llm_kernel_test" / "sandbox" / "generated"
        self.prompt_generator = PromptGenerator(project_root)

    def get_all_definitions(self) -> List[Path]:
        """获取所有非模板的 definition 文件"""
        definitions = []
        for def_path in self.definitions_dir.rglob("*.json"):
            # 跳过模板文件
            if "templates" in str(def_path):
                continue
            # 跳过元数据文件
            if def_path.name == "model_architectures.json":
                continue
            definitions.append(def_path)
        return sorted(definitions)

    def definition_to_attempt_id(self, def_path: Path) -> str:
        """
        将 definition 路径转换为 attempt_id

        命名规范：
        - {model}_{layer}_{variant}
        例如: llama_llama3_8b_att_qkv, qwen3_qwen3_32b_ffn_up

        从路径: definitions/quant_gemm/llama/w8a32c8_q8_0_fp32_int8_llama3_8b_att_qkv_n12288_k4096.json
        提取: model=llama, name中的模型和层信息
        """
        # 从文件名提取信息
        filename = def_path.stem  # 去掉 .json

        # 从路径提取模型类别
        parts = def_path.relative_to(self.definitions_dir).parts
        if len(parts) >= 2:
            category = parts[1]  # e.g., "llama", "qwen3", "deepseek_v3"
        else:
            category = "unknown"

        # 从文件名解析: w8a32c8_q8_0_fp32_int8_{model}_{layer}_{n}_{k}.json
        # 提取模型和层信息
        parts_name = filename.split("_")

        # 跳过量化格式前缀 (w8a32c8_q8_0_fp32_int8_)
        idx = 0
        prefixes = ["w4a16", "w4a8", "w4a8c8", "w5_1a8", "w8a8", "w8a16", "w8a32c8"]
        for prefix in prefixes:
            if filename.startswith(prefix):
                idx = len(prefix.split("_"))
                break

        # 提取模型名和层信息
        # 例如: llama3_8b_att_qkv_n12288_k4096
        model_layer_parts = []
        for part in parts_name[idx:]:
            # 跳过维度部分 (n开头)
            if part.startswith("n") and part[1:].isdigit():
                break
            model_layer_parts.append(part)

        if not model_layer_parts:
            # 使用完整文件名（不含量化前缀）
            model_layer_parts = parts_name[idx:]

        model_layer = "_".join(model_layer_parts)

        # 组合: category_model_layer
        # 但如果 model_layer 已经包含了模型名，就只用 model_layer
        if model_layer.startswith(category):
            attempt_id = model_layer
        else:
            attempt_id = f"{category}_{model_layer}"

        # 清理尝试ID：移除冗余信息，保持简洁
        # 例如: llama_llama3_8b_att_qkv -> llama3_8b_att_qkv
        attempt_id_parts = attempt_id.split("_")
        if len(attempt_id_parts) > 1 and attempt_id_parts[0] == attempt_id_parts[1].split("3")[0].split("2")[0]:
            # 去掉重复的模型前缀
            attempt_id_parts = attempt_id_parts[1:]

        attempt_id = "_".join(attempt_id_parts)

        return attempt_id

    def definition_to_spec(self, def_path: Path) -> Dict:
        """将 definition 转换为 spec 格式"""
        definition = self.prompt_generator.load_definition(def_path)
        return self.prompt_generator._definition_to_spec(definition)

    def create_test_directory(
        self,
        def_path: Path,
        attempt_id: Optional[str] = None,
        kernel_template: Optional[str] = None,
        generate_prompt: bool = True,
        prompt_style: str = "focused"
    ) -> Dict[str, Any]:
        """
        为 definition 创建测试目录

        Args:
            def_path: definition 文件路径
            attempt_id: 指定的 attempt_id（默认自动生成）
            kernel_template: 用作 kernel.cu 模板的 attempt_id
            generate_prompt: 是否生成 prompt.md
            prompt_style: prompt 风格 (full/minimal/focused)

        Returns:
            创建结果字典
        """
        result = {
            "definition": str(def_path),
            "attempt_id": None,
            "test_dir": None,
            "success": False,
            "error": None
        }

        try:
            # 1. 生成 attempt_id
            if attempt_id is None:
                attempt_id = self.definition_to_attempt_id(def_path)
            result["attempt_id"] = attempt_id

            # 2. 转换 definition 为 spec
            spec = self.definition_to_spec(def_path)

            # 3. 确定变体名称（统一使用 w8a32c8_q8_0_fp32_int8）
            variant = "w8a32c8_q8_0_fp32_int8"

            # 4. 创建测试目录
            test_dir = self.sandbox_dir / attempt_id / variant
            test_dir.mkdir(parents=True, exist_ok=True)
            result["test_dir"] = str(test_dir)

            # 5. 保存 spec.json
            spec_file = test_dir / "spec.json"
            # 添加 source_definition 追溯
            spec["source_definition"] = str(def_path.relative_to(self.project_root))
            with open(spec_file, "w") as f:
                json.dump(spec, f, indent=2)

            # 6. 复制或生成 kernel.cu
            kernel_file = test_dir / "kernel.cu"

            if kernel_template:
                # 从模板 attempt 复制 kernel.cu
                # 使用原始 sandbox 目录查找模板（批次模式下）
                template_dir = self.project_root / "llm_kernel_test" / "sandbox" / "generated" / kernel_template
                if template_dir.exists():
                    template_kernel = None
                    # 查找变体目录
                    for v in template_dir.iterdir():
                        if v.is_dir() and (v / "kernel.cu").exists():
                            template_kernel = v / "kernel.cu"
                            break

                    if template_kernel and template_kernel.exists():
                        shutil.copy(template_kernel, kernel_file)
                        result["kernel_source"] = f"copied from {kernel_template}"
                    else:
                        result["error"] = f"Template kernel not found in {kernel_template}"
                        return result
                else:
                    result["error"] = f"Template directory not found: {kernel_template}"
                    return result
            else:
                # 创建占位符 kernel.cu（供 LLM 生成或手动编写）
                placeholder = self._create_placeholder_kernel(spec, attempt_id)
                with open(kernel_file, "w") as f:
                    f.write(placeholder)
                result["kernel_source"] = "placeholder"

            # 7. 生成 prompt（如果需要）
            if generate_prompt:
                prompt = self.prompt_generator.generate_prompt(spec, style=prompt_style)
                prompt_file = test_dir / "prompt.md"
                with open(prompt_file, "w") as f:
                    f.write(prompt)
                result["prompt_file"] = str(prompt_file.relative_to(self.project_root))

            # 8. 创建 metadata.json
            metadata = {
                "attempt_id": attempt_id,
                "variant": variant,
                "created_at": datetime.now().isoformat(),
                "source_definition": str(def_path.relative_to(self.project_root)),
                "kernel_template": kernel_template,
                "prompt_style": prompt_style,
            }
            metadata_file = test_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # 9. 尝试复制 reference.py（如果存在）
            template_variant_dir = self.project_root / "llm_kernel_test" / "templates" / variant
            if template_variant_dir.exists():
                ref_file = template_variant_dir / "reference.py"
                if ref_file.exists():
                    shutil.copy(ref_file, test_dir / "reference.py")
                    result["reference_py"] = "copied from template"

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False

        return result

    def _create_placeholder_kernel(self, spec: Dict, attempt_id: str) -> str:
        """创建占位符 kernel.cu"""
        variant = spec.get("name", "unknown")
        weight_type = spec.get("inputs", {}).get("weight", {}).get("dtype", "block_q8_0")
        activation_type = spec.get("inputs", {}).get("activation", {}).get("dtype", "float32")

        N = spec.get("params", {}).get("N", {}).get("default", 4096)
        K = spec.get("params", {}).get("K", {}).get("default", 4096)

        return f'''// Placeholder Kernel for {attempt_id}
// Variant: {variant}
// Weight: {weight_type}, Activation: {activation_type}
// Target Dimensions: N={N}, K={K}
//
// This kernel was auto-generated from a definition file.
// Please implement the kernel computation or use LLM to generate it.
//
// To generate with LLM:
//   python -m python.tools.definition_to_test --definition <path> --generate-with-llm

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// TODO: Implement kernel for {attempt_id}
// Target dimensions: N={N}, K={K}

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int M, int N, int K
) {{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::zeros({{M, N}}, torch::dtype(torch::kFloat32).device(weight.device()));

    // TODO: Implement GEMM computation

    return output;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &forward, "Placeholder kernel - please implement");
}}
'''

    def create_all_tests(
        self,
        kernel_template: Optional[str] = None,
        generate_prompt: bool = True,
        prompt_style: str = "focused",
        pattern: Optional[str] = None,
        batch_name: Optional[str] = None,
        overwrite: bool = False
    ) -> List[Dict]:
        """
        为所有 definitions 创建测试目录

        Args:
            kernel_template: 用作 kernel 模板的 attempt_id
            generate_prompt: 是否生成 prompt
            prompt_style: prompt 风格
            pattern: 只处理匹配此模式的 definition (glob pattern)
            batch_name: 批次名称（用于创建带时间戳的目录）
            overwrite: 是否覆盖已存在的 attempt 目录
        """
        definitions = self.get_all_definitions()

        if pattern:
            # 过滤匹配的 definitions
            filtered = []
            for def_path in definitions:
                if pattern in str(def_path):
                    filtered.append(def_path)
            definitions = filtered

        # 如果指定了 batch_name，创建批次目录（不覆盖现有测试）
        if batch_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_dir = self.sandbox_dir / f"batch_{timestamp}_{batch_name}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            # 使用批次目录作为新的基础目录
            original_sandbox = self.sandbox_dir
            self.sandbox_dir = batch_dir
            print(f"使用批次目录: {batch_dir.relative_to(self.project_root)}")

        results = []
        print(f"Creating test directories for {len(definitions)} definitions...")
        print(f"Kernel template: {kernel_template or 'placeholder'}")
        print(f"Generate prompt: {generate_prompt}")
        print(f"Overwrite existing: {overwrite}")
        print()

        for i, def_path in enumerate(definitions, 1):
            print(f"[{i}/{len(definitions)}] {def_path.name}")

            # 检查是否已存在
            attempt_id = self.definition_to_attempt_id(def_path)
            test_dir = self.sandbox_dir / attempt_id / "w8a32c8_q8_0_fp32_int8"

            if test_dir.exists() and not overwrite:
                print(f"  ⏭️  跳过（已存在）: {attempt_id}")
                print(f"     使用 --overwrite 强制覆盖")
                results.append({
                    "definition": str(def_path),
                    "attempt_id": attempt_id,
                    "test_dir": str(test_dir),
                    "success": True,
                    "skipped": True,
                    "reason": "already_exists"
                })
                print()
                continue

            result = self.create_test_directory(
                def_path,
                kernel_template=kernel_template,
                generate_prompt=generate_prompt,
                prompt_style=prompt_style
            )

            if result["success"]:
                print(f"  ✅ Created: {result['attempt_id']}")
                print(f"     → {result['test_dir']}")
                if "kernel_source" in result:
                    print(f"     Kernel: {result['kernel_source']}")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

            results.append(result)
            print()

        # 恢复原始 sandbox 目录
        if batch_name:
            self.sandbox_dir = original_sandbox

        return results

    def print_summary(self, results: List[Dict]):
        """打印创建摘要"""
        total = len(results)
        success = sum(1 for r in results if r["success"])

        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total: {total}")
        print(f"Success: {success}")
        print(f"Failed: {total - success}")
        print()

        if success > 0:
            print("Created attempt IDs:")
            for r in results:
                if r["success"]:
                    print(f"  - {r['attempt_id']}")

        print()
        print("Next steps:")
        print("  1. Generate kernels with LLM:")
        print("     python -m python.tools.definition_to_test --all --generate-with-llm")
        print()
        print("  2. Or run tests (if kernels are ready):")
        print("     python -m llm_kernel_test.batch_test_runner --test-all")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="从 Definition 文件创建测试目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 为单个 definition 创建测试
  python -m python.tools.definition_to_test --definition definitions/quant_gemm/llama/...

  # 为所有 definitions 创建测试
  python -m python.tools.definition_to_test --all

  # 使用 deepseek_v3 的 kernel 作为模板
  python -m python.tools.definition_to_test --all --kernel-template deepseek_v3

  # 只处理特定模型的 definitions
  python -m python.tools.definition_to_test --all --pattern qwen3

  # 创建测试并运行
  python -m python.tools.definition_to_test --all --test
        """
    )

    # 输入选项
    parser.add_argument(
        "--definition", "-d",
        type=Path,
        help="单个 definition 文件路径"
    )
    parser.add_argument("--all", action="store_true", help="处理所有 definitions")

    # Kernel 模板选项
    parser.add_argument(
        "--kernel-template", "-k",
        type=str,
        help="用作 kernel.cu 模板的 attempt_id (如: deepseek_v3, baseline)"
    )

    # Prompt 选项
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="不生成 prompt.md"
    )
    parser.add_argument(
        "--prompt-style",
        choices=["full", "minimal", "focused"],
        default="focused",
        help="Prompt 风格 (默认: focused)"
    )

    # 过滤选项
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        help="只处理匹配此模式的 definition"
    )

    # 批次选项
    parser.add_argument(
        "--batch-name", "-b",
        type=str,
        help="批次名称（创建带时间戳的批次目录，避免覆盖）"
    )
    parser.add_argument(
        "--overwrite", "-o",
        action="store_true",
        help="覆盖已存在的 attempt 目录"
    )

    # 输出选项
    parser.add_argument(
        "--attempt-id",
        type=str,
        help="指定 attempt_id（仅用于单个 definition）"
    )

    # 测试选项
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="创建后自动运行测试"
    )

    args = parser.parse_args()

    pipeline = DefinitionToTestPipeline()

    if args.definition:
        # 单个 definition
        result = pipeline.create_test_directory(
            args.definition,
            attempt_id=args.attempt_id,
            kernel_template=args.kernel_template,
            generate_prompt=not args.no_prompt,
            prompt_style=args.prompt_style
        )

        if result["success"]:
            print(f"✅ Created: {result['attempt_id']}")
            print(f"   Directory: {result['test_dir']}")

            if args.test:
                print()
                print("Running tests...")
                # TODO: 调用 test_runner
        else:
            print(f"❌ Failed: {result.get('error')}")
            sys.exit(1)

    elif args.all:
        # 所有 definitions
        results = pipeline.create_all_tests(
            kernel_template=args.kernel_template,
            generate_prompt=not args.no_prompt,
            prompt_style=args.prompt_style,
            pattern=args.pattern,
            batch_name=args.batch_name,
            overwrite=args.overwrite
        )

        pipeline.print_summary(results)

        if args.test:
            print()
            print("Running batch tests...")
            # TODO: 调用 batch_test_runner

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
