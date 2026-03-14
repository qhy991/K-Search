#!/usr/bin/env python3
"""
新算子 Definition Baseline 映射工具

支持 Flash Attention、RMS Norm、TopK 等新算子的 baseline 查询。

Usage:
    from core.tools.new_operators_definition_mapper import NewOperatorsDefinitionMapper

    mapper = NewOperatorsDefinitionMapper()

    # RMS Norm 查询
    baseline = mapper.get_rms_norm_baseline(
        definition_path="definitions/rms_norm/llama/fp32_rms_norm_llama3_8b_hs4096.json",
        hardware="RTX4090",
        batch_size=512
    )

    # TopK 查询
    baseline = mapper.get_topk_baseline(
        definition_path="definitions/topk/llama/fp32_top_k_sampling_llama3_8b_k8.json",
        hardware="A100",
        batch_size=8
    )

    # Flash Attention 查询
    baseline = mapper.get_flash_attn_baseline(
        definition_path="definitions/flash_attention/llama/fp32_flash_attention_llama3_8b_f16_cache512.json",
        hardware="RTX4090",
        batch_size=512
    )
"""
import json
from pathlib import Path
from typing import Dict, Optional


class NewOperatorsDefinitionMapper:
    """新算子 Definition Baseline 映射器"""

    def __init__(self, baseline_dir: str = None):
        if baseline_dir is None:
            baseline_dir = Path(__file__).parent.parent.parent / "data" / "baseline"

        self.baseline_dir = Path(baseline_dir)

        # 加载各算子的 baseline 数据
        self.flash_attn_data = self._load_json("flash_attn_baseline.json")
        self.rms_norm_data = self._load_json("rms_norm_baseline.json")
        self.topk_data = self._load_json("topk_baseline.json")

    def _load_json(self, filename: str) -> Dict:
        """加载 JSON 文件"""
        file_path = self.baseline_dir / filename
        if not file_path.exists():
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_definition(self, definition_path: str) -> Dict:
        """加载 definition 文件"""
        def_path = Path(definition_path)
        if not def_path.exists():
            # 尝试从项目根目录解析
            project_root = Path(__file__).parent.parent.parent
            def_path = project_root / definition_path

        with open(def_path) as f:
            return json.load(f)

    # ==================== RMS Norm ====================

    def get_rms_norm_baseline(
        self,
        definition_path: str,
        hardware: str = "RTX4090",
        batch_size: int = 512
    ) -> Optional[Dict]:
        """
        获取 RMS Norm definition 对应的基线数据

        Args:
            definition_path: definition 文件路径
            hardware: 硬件名称
            batch_size: batch size

        Returns:
            基线数据字典
        """
        definition = self._load_definition(definition_path)

        hidden_size = definition["axes"]["hidden_size"]["value"]
        ne = [hidden_size, 1, batch_size, 1]

        # 构建 case_id
        ne_str = "x".join(map(str, ne))
        case_id = f"rms_norm_hs{hidden_size}_{ne_str}"

        return self._get_baseline("rms_norm", case_id, hardware, definition, batch_size)

    # ==================== TopK ====================

    def get_topk_baseline(
        self,
        definition_path: str,
        hardware: str = "RTX4090",
        batch_size: int = 512
    ) -> Optional[Dict]:
        """
        获取 TopK definition 对应的基线数据

        Args:
            definition_path: definition 文件路径
            hardware: 硬件名称
            batch_size: batch size

        Returns:
            基线数据字典
        """
        definition = self._load_definition(definition_path)

        k = definition["axes"]["k"]["value"]
        vocab_size = 256  # GGML baseline 使用的子词汇表大小
        ne = [vocab_size, batch_size, 1, 1]

        # 构建 case_id
        ne_str = "x".join(map(str, ne))
        case_id = f"topk_k{k}_ne0{vocab_size}_{ne_str}"

        return self._get_baseline("topk", case_id, hardware, definition, batch_size)

    # ==================== Flash Attention ====================

    def get_flash_attn_baseline(
        self,
        definition_path: str,
        hardware: str = "RTX4090",
        batch_size: int = 512
    ) -> Optional[Dict]:
        """
        获取 Flash Attention definition 对应的基线数据

        Args:
            definition_path: definition 文件路径
            hardware: 硬件名称
            batch_size: batch size (nb)

        Returns:
            基线数据字典
        """
        definition = self._load_definition(definition_path)

        # 从描述中提取参数
        # 例如: "fp32_flash_attention_llama3_8b_f16_cache512"
        name = definition["name"]
        parts = name.split("_")

        # 提取模型名称
        model = None
        for part in parts:
            if "llama" in part.lower():
                model = "Llama3-8B"
            elif "qwen" in part.lower() and "7b" in part.lower():
                model = "Qwen2.5-7B"

        # 提取 KV 类型
        kv_type = "F16"  # 默认
        for part in parts:
            if "f16" in part.lower():
                kv_type = "F16"
            elif "q8_0" in part.lower():
                kv_type = "Q8_0"
            elif "q4_0" in part.lower():
                kv_type = "Q4_0"

        # 提取 cache size
        cache_size = None
        for part in parts:
            if "cache" in part.lower():
                cache_size = int(part.lower().replace("cache", ""))

        if not model or not cache_size:
            return {
                "status": "error",
                "message": f"无法从 name 中提取参数: {name}"
            }

        # 构建 case_id
        case_id = f"flash_attn_{model}_{kv_type}_cache{cache_size}_nb{batch_size}"

        return self._get_baseline("flash_attention", case_id, hardware, definition, batch_size)

    # ==================== 通用方法 ====================

    def _get_baseline(
        self,
        op_type: str,
        case_id: str,
        hardware: str,
        definition: Dict,
        batch_size: int
    ) -> Optional[Dict]:
        """从指定算子类型获取 baseline"""
        data_map = {
            "rms_norm": self.rms_norm_data,
            "topk": self.topk_data,
            "flash_attention": self.flash_attn_data
        }

        data = data_map.get(op_type, {})

        if case_id not in data:
            # 查找最接近的配置
            return self._find_closest_baseline(op_type, definition, hardware, batch_size)

        baseline = data[case_id].get("hardware", {}).get(hardware)

        if not baseline:
            return {
                "status": "no_hardware",
                "message": f"硬件 {hardware} 没有此配置的 baseline 数据",
                "case_id": case_id
            }

        return {
            "status": "exact_match",
            "definition_name": definition["name"],
            "op_type": op_type,
            "case_id": case_id,
            "hardware": hardware,
            "batch_size": batch_size,
            "baseline": baseline
        }

    def _find_closest_baseline(
        self,
        op_type: str,
        definition: Dict,
        hardware: str,
        batch_size: int
    ) -> Dict:
        """查找最接近的 baseline 配置"""
        data_map = {
            "rms_norm": self.rms_norm_data,
            "topk": self.topk_data,
            "flash_attention": self.flash_attn_data
        }

        data = data_map.get(op_type, {})
        candidates = []

        for case_id, case_data in data.items():
            hw_data = case_data.get("hardware", {}).get(hardware)
            if not hw_data:
                continue

            # 计算相似度（简化版）
            candidates.append({
                "case_id": case_id,
                "baseline": hw_data,
                "case_data": case_data
            })

        if candidates:
            # 按某种规则排序，这里简单返回第一个
            closest = candidates[0]
            return {
                "status": "closest_match",
                "definition_name": definition["name"],
                "op_type": op_type,
                "hardware": hardware,
                "batch_size": batch_size,
                "closest_case_id": closest["case_id"],
                "baseline": closest["baseline"]
            }

        return {
            "status": "no_match",
            "message": f"未找到 {op_type} 的 baseline 数据"
        }

    # ==================== 批量查询 ====================

    def batch_compare(self, definitions_dir: str, hardware: str = "RTX4090") -> list:
        """
        批量查询所有新算子 definitions 的 baseline

        Args:
            definitions_dir: definitions 目录
            hardware: 硬件名称

        Returns:
            比较结果列表
        """
        results = []
        defs_path = Path(definitions_dir)

        # 支持的算子类型
        op_type_dirs = {
            "rms_norm": self.get_rms_norm_baseline,
            "topk": self.get_topk_baseline,
            "flash_attention": self.get_flash_attn_baseline
        }

        for json_file in defs_path.rglob("*.json"):
            try:
                # 根据文件路径确定算子类型
                op_type = None
                get_baseline_func = None

                for ot, func in op_type_dirs.items():
                    if ot in str(json_file):
                        op_type = ot
                        get_baseline_func = func
                        break

                if not get_baseline_func:
                    continue

                # 调用对应的 baseline 查询函数
                baseline = get_baseline_func(str(json_file), hardware, 512)

                result = {
                    "definition": json_file.name,
                    "op_type": op_type,
                    "definition_name": baseline.get("definition_name", json_file.stem),
                    "status": baseline.get("status"),
                    "hardware": hardware
                }

                if baseline["status"] == "exact_match":
                    metric_key = "tflops" if op_type == "flash_attention" else "gbps"
                    result["baseline"] = baseline["baseline"][metric_key]

                results.append(result)

            except Exception as e:
                results.append({
                    "definition": json_file.name,
                    "op_type": op_type,
                    "error": str(e)
                })

        return results


# 便捷函数
def get_new_operator_baseline(
    definition_path: str,
    hardware: str = "RTX4090",
    batch_size: int = 512
) -> Optional[Dict]:
    """快速获取新算子 definition 的 baseline"""
    mapper = NewOperatorsDefinitionMapper()

    # 根据 definition 路径判断算子类型
    if "rms_norm" in definition_path:
        return mapper.get_rms_norm_baseline(definition_path, hardware, batch_size)
    elif "topk" in definition_path:
        return mapper.get_topk_baseline(definition_path, hardware, batch_size)
    elif "flash_attention" in definition_path:
        return mapper.get_flash_attn_baseline(definition_path, hardware, batch_size)
    else:
        return {"status": "unknown", "message": "未知的算子类型"}


if __name__ == "__main__":
    # 测试示例
    mapper = NewOperatorsDefinitionMapper()

    print("=" * 60)
    print("新算子 Baseline 映射测试")
    print("=" * 60)

    # 测试 RMS Norm
    print("\n1. RMS Norm")
    result = mapper.get_rms_norm_baseline(
        "definitions/rms_norm/llama/fp32_rms_norm_llama3_8b_hs4096.json",
        "RTX4090",
        512
    )
    print(f"   Status: {result['status']}")
    if result['status'] == 'exact_match':
        print(f"   GB/s: {result['baseline']['gbps']}")
        print(f"   Latency: {result['baseline']['us_per_run']} us")

    # 测试 TopK
    print("\n2. TopK")
    result = mapper.get_topk_baseline(
        "definitions/topk/llama/fp32_top_k_sampling_llama3_8b_k8.json",
        "RTX4090",
        512
    )
    print(f"   Status: {result['status']}")
    if result['status'] == 'exact_match':
        print(f"   GB/s: {result['baseline']['gbps']}")

    # 测试 Flash Attention
    print("\n3. Flash Attention")
    result = mapper.get_flash_attn_baseline(
        "definitions/flash_attention/llama/fp32_flash_attention_llama3_8b_f16_cache512.json",
        "RTX4090",
        512
    )
    print(f"   Status: {result['status']}")
    if result['status'] == 'exact_match':
        print(f"   TFLOPS: {result['baseline']['tflops']}")

    print("\n" + "=" * 60)
