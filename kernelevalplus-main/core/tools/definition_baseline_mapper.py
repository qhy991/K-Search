#!/usr/bin/env python3
"""
Definition 与 Baseline 映射工具

提供正确的维度映射，将 definition 的配置映射到 GGML baseline case_id。

## Case ID 格式

### Definition Case ID (完整，包含模型/层):
w{weight}a{activation}c{compute}_{type_a}_{type_b}_{model}_{layer}_n{N}_k{K}
例如: w4a32c8_q4_0_fp32_int8_ds3_lm_head_n129280_k7168

### GGML Baseline Case ID (仅维度):
w{weight}a{activation}c{compute}_{type_a}_{type_b}_m{M}_n{N}_k{K}
例如: w4a32c8_q4_0_f32_m129280_n1_k7168

## 映射关系

| 字段 | 来源 | 示例 |
|------|------|------|
| w{weight} | variant 标签 | w4, w8 |
| a{activation} | variant 标签 | a32, a16 |
| c{compute} | variant 标签 | c8 |
| type_a | weight dtype | q4_0, q8_0 |
| type_b | activation dtype | f32, q8_1 |
| model | model_architectures | ds3, llama3_8b |
| layer | tags 或文件名 | lm_head, att_qkv |
| N, K | axes 值 | n129280, k7168 |

Usage:
    from core.tools.definition_baseline_mapper import DefinitionBaselineMapper

    mapper = DefinitionBaselineMapper()

    # 获取 definition 对应的基线数据
    baseline = mapper.get_baseline(
        definition_path="definitions/quant_gemm/llama/w8a32c8_q8_0_fp32_int8_llama3_8b_att_qkv_n12288_k4096.json",
        hardware="RTX4090",
        batch_size=1
    )

    # 获取 definition case_id
    def_case_id = mapper.get_definition_case_id(definition_path)
    # => "w8a32c8_q8_0_fp32_int8_llama3_8b_att_qkv_n12288_k4096"
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, List

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ggml_baseline_api import BaselineAPI


class DefinitionBaselineMapper:
    """Definition 与 Baseline 映射器"""

    # 模型简称映射
    MODEL_SHORT_NAMES = {
        "deepseek-v2": "ds2",
        "deepseek_v2": "ds2",
        "deepseek-v3": "ds3",
        "deepseek_v3": "ds3",
        "llama-3-8b": "llama3_8b",
        "llama3-8b": "llama3_8b",
        "llama3_8b": "llama3_8b",
        "llama-3-70b": "llama3_70b",
        "llama3-70b": "llama3_70b",
        "llama3_70b": "llama3_70b",
        "llama2-7b": "llama2_7b",
        "mistral-7b": "mistral7b",
        "mixtral-8x7b": "mixtral8x7b",
        "qwen2.5-7b": "qwen2_5_7b",
        "qwen-2.5-7b": "qwen2_5_7b",
        "qwen2-5-7b": "qwen2_5_7b",
        "qwen2_5_7b": "qwen2_5_7b",
        "qwen3-0.6b": "qwen3_0_6b",
        "qwen3-1.5b": "qwen3_1_5b",
        "qwen3-4b": "qwen3_4b",
        "qwen3-8b": "qwen3_8b",
        "qwen3-14b": "qwen3_14b",
        "qwen3-32b": "qwen3_32b",
    }

    # 层类型映射
    LAYER_TYPE_NAMES = {
        "lm-head": "lm_head",
        "lm_head": "lm_head",
        "att_out": "att_out",
        "att-qkv": "att_qkv",
        "att_qkv": "att_qkv",
        "attention-qkv": "att_qkv",  # 新增
        "attention": "att",
        "att": "att",
        "ffn_up": "ffn_up",
        "ffn-down": "ffn_down",
        "ffn_down": "ffn_down",
        "ffn": "ffn",
        "moe_up": "moe_up",
        "moe-down": "moe_down",
        "moe_down": "moe_down",
        "moe_routing_up": "moe_routing_up",
        "moe_routing_down": "moe_routing_down",
        "gate": "gate",
        "router": "router",
    }

    def __init__(self, baseline_path: str = None):
        self.api = BaselineAPI(baseline_path)

        # 维度对应关系
        # GGML: C(M,N) = A(M,K) @ W(N,K)^T
        #  Definition: C(M,N) = A(M,K) @ W(N,K)^T
        #
        # 关键区别：
        # - GGML 的 M 是输出特征维度 (对应我们的 N)
        # - GGML 的 N 是 batch size (对应我们的 M)
        # - GGML 的 K 是输入特征维度 (对应我们的 K)

    def _extract_definition_info(self, definition_path: str) -> Dict:
        """从 definition 文件提取信息"""
        def_path = Path(definition_path)

        with open(def_path) as f:
            definition = json.load(f)

        # 获取维度
        n = definition["axes"]["N"]["value"]
        k = definition["axes"]["K"]["value"]

        # 获取量化类型
        variant = definition.get("variant", "")
        if "W8A32C8" in variant:
            type_a = "q8_0"
        elif "W4A32C8" in variant:
            type_a = "q4_0"
        elif "W4A16" in variant:
            type_a = "q4_0"  # 近似
        elif "W4A8C8" in variant:
            type_a = "q4_0"
        elif "W4A8C16" in variant:
            type_a = "q4_0"
        elif "W5_1A8" in variant:
            type_a = "q5_1"
        elif "W8A8" in variant:
            type_a = "q8_0"
        else:
            type_a = None

        # 获取激活类型
        activation_dtype = definition["inputs"]["activation"]["dtype"]
        if activation_dtype == "float32":
            type_b_storage = "f32"
        elif activation_dtype == "float16":
            type_b_storage = "f16"
        else:
            type_b_storage = activation_dtype

        # 获取激活量化类型 (用于 compute pattern，如 Q8_1)
        # 这是从 weight dtype 推断的 llama.cpp 计算模式
        weight_dtype = definition["inputs"]["weight"]["dtype"]
        if weight_dtype == "block_q4_0":
            type_b_compute = "q8_1"  # Q4_0 × Q8_1 pattern
        elif weight_dtype == "block_q4_1":
            type_b_compute = "q8_1"  # Q4_1 × Q8_1 pattern
        elif weight_dtype == "block_q8_0":
            type_b_compute = "q8_0"  # Q8_0 × Q8_0 pattern
        else:
            type_b_compute = type_b_storage  # 回退到存储类型

        # 获取模型信息
        model_architectures = definition.get("model_architectures", [])
        if model_architectures:
            model_key = model_architectures[0].lower()
            # 尝试直接匹配
            if model_key in self.MODEL_SHORT_NAMES:
                model = self.MODEL_SHORT_NAMES[model_key]
            else:
                # 尝试替换下划线为连字符后匹配
                model_key_alt = model_key.replace("_", "-")
                if model_key_alt in self.MODEL_SHORT_NAMES:
                    model = self.MODEL_SHORT_NAMES[model_key_alt]
                else:
                    # 最后尝试：替换连字符为下划线，移除版本号中的点
                    fallback = model_key.replace("-", "_").replace(".", "_")
                    model = self.MODEL_SHORT_NAMES.get(model_key_alt, fallback)
                    # 如果还是没有匹配，尝试更激进的清理
                    if model == fallback:
                        # 移除所有非字母数字字符，保留下划线
                        import re
                        cleaned = re.sub(r'[^a-z0-9_]', '_', model_key)
                        # 移除多余的下划线
                        cleaned = re.sub(r'_+', '_', cleaned)
                        model = cleaned
        else:
            model = "unknown"

        # 获取层类型 (从 tags 中提取)
        layer = "unknown"
        for tag in definition.get("tags", []):
            if tag.startswith("layer:"):
                layer_key = tag.split(":", 1)[1].lower()
                layer = self.LAYER_TYPE_NAMES.get(layer_key, layer_key.replace("-", "_"))
                break

        # 如果没有从 tags 获取到，尝试从 description 推断
        if layer == "unknown" or layer == "att":
            desc = definition.get("description", "").lower()
            name = definition.get("name", "").lower()

            # 更精确的层类型推断
            if "lm head" in desc or "lm_head" in desc or "lm_head" in name:
                layer = "lm_head"
            elif "att_qkv" in name or ("qkv" in name and "att" in name):
                layer = "att_qkv"
            elif "att_out" in name or ("output" in name and "att" in name):
                layer = "att_out"
            elif "ffn_up" in name or ("ffn" in name and "up" in name):
                layer = "ffn_up"
            elif "ffn_down" in name or ("ffn" in name and "down" in name):
                layer = "ffn_down"
            elif "moe_up" in name or ("moe" in name and "up" in name):
                layer = "moe_up"
            elif "moe_down" in name or ("moe" in name and "down" in name):
                layer = "moe_down"
            elif "moe_routing" in name:
                if "up" in name:
                    layer = "moe_routing_up"
                elif "down" in name:
                    layer = "moe_routing_down"

        return {
            "name": definition.get("name", def_path.stem),
            "variant": variant,
            "type_a": type_a,
            "type_b_storage": type_b_storage,
            "type_b_compute": type_b_compute,
            "n": n,
            "k": k,
            "model": model,
            "layer": layer
        }

    def _build_baseline_case_id(self, type_a: str, n: int, k: int, batch_size: int = 1, type_b: str = "f32") -> str:
        """
        构建对应的 baseline case_id

        格式: w{weight}a{activation}c{compute}_{type_a}_{type_b}_m{M}_n{N}_k{K}
        例如: w4a32c8_q4_0_f32_m4096_n1_k4096

        Args:
            type_a: 量化类型 (q4_0, q8_0 等)
            n: 输出特征维度 (对应 GGML 的 M)
            k: 输入特征维度 (对应 GGML 的 K)
            batch_size: batch size (对应 GGML 的 N)
            type_b: 激活类型 (默认 f32)
        """
        # 获取位宽
        if type_a.startswith("q4"):
            w_bits = 4
        elif type_a.startswith("q5"):
            w_bits = 5
        elif type_a.startswith("q8"):
            w_bits = 8
        else:
            w_bits = 8

        # 激活位宽
        if type_b == "f32" or type_b == "float32":
            a_bits = 32
        elif type_b == "f16" or type_b == "float16" or type_b == "fp16":
            a_bits = 16
        elif type_b.startswith("q8"):
            a_bits = 8
        else:
            a_bits = 32

        # 计算位宽 (GGML 都是 INT8 DP4A)
        c_bits = 8

        return f"w{w_bits}a{a_bits}c{c_bits}_{type_a}_{type_b}_m{n}_n{batch_size}_k{k}"

    def get_definition_case_id(self, definition_path: str) -> str:
        """
        获取 definition 的完整 case_id（包含模型和层信息）

        格式: w{weight}a{activation}c{compute}_{type_a}_{type_b}_{model}_{layer}_n{N}_k{K}
        例如: w4a32c8_q4_0_fp32_int8_ds3_lm_head_n129280_k7168

        Args:
            definition_path: definition 文件路径

        Returns:
            Definition case_id 字符串
        """
        info = self._extract_definition_info(definition_path)

        # 从 variant 获取位宽
        variant = info["variant"]
        if "W4A32C8" in variant:
            w_bits, a_bits, c_bits = 4, 32, 8
        elif "W4A16C16" in variant:
            w_bits, a_bits, c_bits = 4, 16, 16
        elif "W8A32C8" in variant:
            w_bits, a_bits, c_bits = 8, 32, 8
        elif "W4A8C8" in variant:
            w_bits, a_bits, c_bits = 4, 8, 8
        elif "W5_1A8" in variant:
            w_bits, a_bits, c_bits = 5, 8, 8
        elif "W8A8" in variant:
            w_bits, a_bits, c_bits = 8, 8, 8
        else:
            w_bits, a_bits, c_bits = 8, 32, 8  # 默认

        type_a = info["type_a"] or "q8_0"
        type_b_compute = info["type_b_compute"]

        return f"w{w_bits}a{a_bits}c{c_bits}_{type_a}_{type_b_compute}_{info['model']}_{info['layer']}_n{info['n']}_k{info['k']}"

    def get_baseline(
        self,
        definition_path: str,
        hardware: str = "RTX4090",
        batch_size: int = 1
    ) -> Optional[Dict]:
        """
        获取 definition 对应的基线数据

        Args:
            definition_path: definition 文件路径
            hardware: 硬件名称
            batch_size: batch size (默认 1)

        Returns:
            基线数据字典，包含原始配置和性能数据
        """
        info = self._extract_definition_info(definition_path)

        if info["type_a"] is None:
            return {
                "status": "unsupported_type",
                "message": f"不支持的变体类型: {info['variant']}"
            }

        # 生成 definition case_id（包含模型和层）
        definition_case_id = self.get_definition_case_id(definition_path)

        # 构建 baseline case_id
        # 我们的 N (output) → GGML 的 M
        # 我们的 M (batch) → GGML 的 N
        # 注意：baseline 使用存储类型 (f32)，而非计算类型 (q8_1)
        baseline_case_id = self._build_baseline_case_id(
            info["type_a"], info["n"], info["k"], batch_size, info["type_b_storage"]
        )

        # 查询基线
        baseline = self.api.get_by_case_id(baseline_case_id, hardware)

        if baseline:
            return {
                "status": "exact_match",
                "definition_case_id": definition_case_id,
                "definition_name": info["name"],
                "definition_variant": info["variant"],
                "model": info["model"],
                "layer": info["layer"],
                "type_a": info["type_a"],
                "type_b_storage": info["type_b_storage"],
                "type_b_compute": info["type_b_compute"],
                "our_dims": {"M": batch_size, "N": info["n"], "K": info["k"]},
                "baseline_case_id": baseline_case_id,
                "baseline_dims": {"M": info["n"], "N": batch_size, "K": info["k"]},
                "hardware": hardware,
                "baseline": baseline
            }

        # 没有精确匹配，查找最接近的
        closest = self.api.get_closest_baseline(
            hardware, info["type_a"], info["n"], batch_size, info["k"], info["type_b_storage"]
        )

        if closest:
            return {
                "status": "closest_match",
                "definition_case_id": definition_case_id,
                "definition_name": info["name"],
                "definition_variant": info["variant"],
                "model": info["model"],
                "layer": info["layer"],
                "type_a": info["type_a"],
                "type_b_storage": info["type_b_storage"],
                "type_b_compute": info["type_b_compute"],
                "our_dims": {"M": batch_size, "N": info["n"], "K": info["k"]},
                "baseline_case_id": baseline_case_id,
                "baseline_dims": {"M": info["n"], "N": batch_size, "K": info["k"]},
                "hardware": hardware,
                "closest": closest
            }

        return {
            "status": "no_match",
            "definition_case_id": definition_case_id,
            "definition_name": info["name"],
            "definition_variant": info["variant"],
            "model": info["model"],
            "layer": info["layer"],
            "type_a": info["type_a"],
            "type_b_storage": info["type_b_storage"],
            "type_b_compute": info["type_b_compute"],
            "our_dims": {"M": batch_size, "N": info["n"], "K": info["k"]},
            "message": f"未找到匹配的基线数据"
        }

    def compare_with_baseline(
        self,
        definition_path: str,
        actual_tflops: float,
        hardware: str = "RTX4090",
        batch_size: int = 1
    ) -> Dict:
        """
        比较实际性能与基线

        Args:
            definition_path: definition 文件路径
            actual_tflops: 实际测量的 TFLOPS
            hardware: 硬件名称
            batch_size: batch size

        Returns:
            比较结果字典
        """
        baseline_result = self.get_baseline(definition_path, hardware, batch_size)

        if baseline_result["status"] == "exact_match":
            baseline = baseline_result["baseline"]
            ratio = actual_tflops / baseline["tflops"]

            return {
                "status": "compared",
                "definition_name": baseline_result["definition_name"],
                "hardware": hardware,
                "baseline_tflops": baseline["tflops"],
                "actual_tflops": actual_tflops,
                "ratio": ratio,
                "percentage": ratio * 100,
                "better": ratio >= 1.0,
                "diff_percent": (ratio - 1.0) * 100
            }

        elif baseline_result["status"] == "closest_match":
            closest = baseline_result["closest"]
            ratio = actual_tflops / closest["tflops"]

            return {
                "status": "compared_with_closest",
                "definition_name": baseline_result["definition_name"],
                "hardware": hardware,
                "baseline_tflops": closest["tflops"],
                "baseline_case_id": closest["case_id"],
                "baseline_dims": closest,
                "actual_tflops": actual_tflops,
                "ratio": ratio,
                "percentage": ratio * 100,
                "better": ratio >= 1.0,
                "diff_percent": (ratio - 1.0) * 100,
                "note": "使用最接近的基线配置"
            }

        else:
            return {
                "status": "no_baseline",
                "message": baseline_result.get("message", "未找到基线数据")
            }

    def batch_compare(
        self,
        definitions_dir: str,
        hardware: str = "RTX4090",
        exclude_templates: bool = True
    ) -> List[Dict]:
        """
        批量比较所有 definitions

        Args:
            definitions_dir: definitions 目录
            hardware: 硬件名称
            exclude_templates: 是否排除模板文件 (默认 True)

        Returns:
            比较结果列表
        """
        results = []
        defs_path = Path(definitions_dir)

        for json_file in defs_path.rglob("*.json"):
            # 跳过模板文件
            if exclude_templates and "templates" in str(json_file):
                continue

            try:
                baseline = self.get_baseline(str(json_file), hardware)

                if baseline["status"] == "exact_match":
                    results.append({
                        "definition": json_file.name,
                        "definition_case_id": baseline.get("definition_case_id"),
                        "model": baseline.get("model"),
                        "layer": baseline.get("layer"),
                        "match_type": "exact",
                        "baseline_tflops": baseline["baseline"]["tflops"],
                        "hardware": hardware
                    })
                elif baseline["status"] == "closest_match":
                    results.append({
                        "definition": json_file.name,
                        "definition_case_id": baseline.get("definition_case_id"),
                        "model": baseline.get("model"),
                        "layer": baseline.get("layer"),
                        "match_type": "closest",
                        "baseline_tflops": baseline["closest"]["tflops"],
                        "closest_case_id": baseline["closest"]["case_id"],
                        "hardware": hardware
                    })
                else:
                    results.append({
                        "definition": json_file.name,
                        "definition_case_id": baseline.get("definition_case_id"),
                        "model": baseline.get("model"),
                        "layer": baseline.get("layer"),
                        "match_type": "none",
                        "hardware": hardware
                    })
            except Exception as e:
                results.append({
                    "definition": json_file.name,
                    "error": str(e)
                })

        return results


# 便捷函数
def get_baseline_for_definition(
    definition_path: str,
    hardware: str = "RTX4090",
    batch_size: int = 1
) -> Optional[Dict]:
    """快速获取 definition 的基线数据"""
    mapper = DefinitionBaselineMapper()
    return mapper.get_baseline(definition_path, hardware, batch_size)


if __name__ == "__main__":
    # 测试示例
    mapper = DefinitionBaselineMapper()

    print("=" * 70)
    print("Definition Baseline 映射测试")
    print("=" * 70)

    # 测试几个 definitions
    project_root = Path(__file__).parent.parent.parent
    test_defs = [
        project_root / "definitions/quant_gemm/llama/w8a32c8_q8_0_fp32_int8_llama3_8b_att_qkv_n12288_k4096.json",
        project_root / "definitions/quant_gemm/llama/w4a32c8_q4_0_fp32_int8_llama3_8b_ffn_down_n4096_k14336.json",
        project_root / "definitions/quant_gemm/deepseek_v3/w8a32c8_q8_0_fp32_int8_ds3_att_out_n7168_k7168.json",
    ]

    for def_path in test_defs:
        result = mapper.get_baseline(def_path, "RTX4090", 1)

        print(f"\n{result['definition_name']}")
        print(f"  Definition Case ID: {result.get('definition_case_id', 'N/A')}")
        print(f"  模型: {result.get('model', 'N/A')}, 层: {result.get('layer', 'N/A')}")
        print(f"  变体: {result['definition_variant']}")
        print(f"  我们的维度: M={result['our_dims']['M']}, N={result['our_dims']['N']}, K={result['our_dims']['K']}")

        if result["status"] == "exact_match":
            print(f"  ✅ 精确匹配基线")
            print(f"  基线 Case ID: {result['baseline_case_id']}")
            print(f"  基线性能: {result['baseline']['tflops']} TFLOPS")
        elif result["status"] == "closest_match":
            print(f"  ⚠️  使用最接近的基线")
            print(f"  最接近: {result['closest']['case_id']}")
            print(f"  基线性能: {result['closest']['tflops']} TFLOPS")
        else:
            print(f"  ❌ {result.get('message', '未找到匹配')}")
