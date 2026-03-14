#!/usr/bin/env python3
"""
统一 Baseline 查询 API

支持所有算子类型的基线数据查询：
- GEMM (quant_gemm)
- Flash Attention (flash_attention)
- RMS Norm (rms_norm)
- TopK (topk)

使用方法:
    from core.tools.baseline_api import get_baseline_api

    api = get_baseline_api()  # 单例，避免重复加载 JSON

    # GEMM
    api.get_gemm("RTX4090", "q4_0", 4096, 1, 4096)

    # Flash Attention
    api.get_flash_attn("RTX4090", "Llama3-8B", "F16", 512, 512)

    # RMS Norm
    api.get_rms_norm("RTX4090", 4096, [4096, 512, 1, 1])

    # TopK
    api.get_topk("RTX4090", 8, 256, [256, 512, 1, 1])
"""
import json
from pathlib import Path
from typing import Dict, Optional, List


class BaselineAPI:
    """
    统一 Baseline 查询 API

    支持的算子类型：
    - gemm / quant_gemm: 量化矩阵乘法
    - flash_attention: Flash Attention
    - rms_norm: RMS 归一化
    - topk: TopK 采样
    """

    # 算子类型到数据文件的映射
    OPERATOR_FILES = {
        "gemm": "baseline_data_compact.json",
        "quant_gemm": "baseline_data_compact.json",
        "flash_attention": "flash_attn_baseline.json",
        "rms_norm": "rms_norm_baseline.json",
        "topk": "topk_baseline.json",
    }

    # 性能指标类型
    OPERATOR_METRICS = {
        "gemm": "tflops",
        "quant_gemm": "tflops",
        "flash_attention": "tflops",
        "rms_norm": "gbps",
        "topk": "gbps",
    }

    def __init__(self, baseline_dir: str = None):
        """
        初始化 API

        Args:
            baseline_dir: baseline 数据目录路径
        """
        if baseline_dir is None:
            baseline_dir = Path(__file__).parent.parent.parent / "data" / "baseline"

        self.baseline_dir = Path(baseline_dir)

        # 加载所有 baseline 数据
        self._data = {}
        self._load_all_data()

    def _load_all_data(self):
        """加载所有 baseline 数据文件"""
        for op_type, filename in self.OPERATOR_FILES.items():
            if op_type in ["gemm", "quant_gemm"]:
                # GEMM 数据只加载一次
                if "gemm" not in self._data:
                    self._data["gemm"] = self._load_json(filename)
                    self._data["quant_gemm"] = self._data["gemm"]
            else:
                self._data[op_type] = self._load_json(filename)

    def _load_json(self, filename: str) -> Dict:
        """加载 JSON 文件"""
        file_path = self.baseline_dir / filename
        if not file_path.exists():
            return {}
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ==================== GEMM ====================

    def get_gemm(
        self,
        hardware: str,
        type_a: str,
        m: int,
        n: int,
        k: int,
        type_b: str = "f32"
    ) -> Optional[Dict]:
        """
        查询 GEMM baseline

        Args:
            hardware: 硬件名称 (RTX4090, A100, RTX4070)
            type_a: 权重量化类型 (q4_0, q8_0, q4_1, q8_1)
            m: 输出特征维度 (GGML 的 M)
            n: batch size (GGML 的 N)
            k: 输入特征维度
            type_b: 激活类型 (f32, f16)

        Returns:
            {'tflops': float, 'gflops': float, 'latency_ms': float}
        """
        # 构建 case_id
        type_b_clean = type_b.replace("float", "f")
        case_id = f"w{4 if 'q4' in type_a else 8}a32c8_{type_a}_{type_b_clean}_m{m}_n{n}_k{k}"

        data = self._data.get("gemm", {}).get(case_id, {})
        hw_data = data.get("hardware", {}).get(hardware)

        if hw_data:
            return {
                "tflops": hw_data.get("tflops"),
                "gflops": hw_data.get("gflops"),
                "latency_ms": hw_data.get("us_per_run", 0) / 1000 if hw_data.get("us_per_run") else None
            }
        return None

    def get_gemm_closest(
        self,
        hardware: str,
        type_a: str,
        m: int,
        n: int,
        k: int,
        type_b: str = "f32"
    ) -> Optional[Dict]:
        """查找最接近的 GEMM baseline"""
        # 简化实现：遍历查找
        best_match = None
        best_score = float('inf')

        for case_id, case_data in self._data.get("gemm", {}).items():
            if type_a not in case_id:
                continue

            hw_data = case_data.get("hardware", {}).get(hardware)
            if not hw_data:
                continue

            # 解析维度
            dims = case_data.get("dimensions", {})
            m_val = dims.get("M", dims.get("m", 0))
            n_val = dims.get("N", dims.get("n", 0))
            k_val = dims.get("K", dims.get("k", 0))

            # 计算距离
            score = abs(m_val - m) + abs(n_val - n) + abs(k_val - k)

            if score < best_score:
                best_score = score
                best_match = {
                    "case_id": case_id,
                    "m": m_val,
                    "n": n_val,
                    "k": k_val,
                    "tflops": hw_data.get("tflops"),
                    "gflops": hw_data.get("gflops")
                }

        return best_match

    # ==================== Flash Attention ====================

    def get_flash_attn(
        self,
        hardware: str,
        model: str,
        kv_type: str,
        cache_size: int,
        batch_size: int
    ) -> Optional[Dict]:
        """
        查询 Flash Attention baseline

        Args:
            hardware: 硬件名称 (A100, RTX4090, RTX4070)
            model: 模型名称 (Llama3-8B, Qwen2.5-7B)
            kv_type: KV 缓存类型 (F16, Q4_0, Q8_0)
            cache_size: KV 缓存大小 (512, 4096, 8192)
            batch_size: 块数量 (nb)

        Returns:
            {'tflops': float, 'us_per_run': float}
        """
        case_id = f"flash_attn_{model}_{kv_type}_cache{cache_size}_nb{batch_size}"
        data = self._data.get("flash_attention", {}).get(case_id, {})
        hw_data = data.get("hardware", {}).get(hardware)

        if hw_data:
            return {
                "tflops": hw_data.get("tflops"),
                "us_per_run": hw_data.get("us_per_run")
            }
        return None

    def get_flash_attn_models(self) -> List[str]:
        """获取所有支持的 Flash Attention 模型"""
        models = set()
        for case_data in self._data.get("flash_attention", {}).values():
            model = case_data.get("model")
            if model:
                models.add(model)
        return sorted(list(models))

    # ==================== RMS Norm ====================

    def get_rms_norm(
        self,
        hardware: str,
        hidden_size: int,
        ne: List[int]
    ) -> Optional[Dict]:
        """
        查询 RMS Norm baseline

        Args:
            hardware: 硬件名称
            hidden_size: 隐藏层维度
            ne: 张量形状 [ne0, ne1, ne2, ne3]

        Returns:
            {'gbps': float, 'us_per_run': float}
        """
        ne_str = "x".join(map(str, ne))
        case_id = f"rms_norm_hs{hidden_size}_{ne_str}"
        data = self._data.get("rms_norm", {}).get(case_id, {})
        hw_data = data.get("hardware", {}).get(hardware)

        if hw_data:
            return {
                "gbps": hw_data.get("gbps"),
                "us_per_run": hw_data.get("us_per_run")
            }
        return None

    def get_rms_norm_hidden_sizes(self) -> List[int]:
        """获取所有支持的 RMS Norm hidden_size"""
        sizes = set()
        for case_data in self._data.get("rms_norm", {}).values():
            size = case_data.get("hidden_size")
            if size:
                sizes.add(size)
        return sorted(list(sizes))

    # ==================== TopK ====================

    def get_topk(
        self,
        hardware: str,
        k: int,
        ne0: int,
        ne: List[int]
    ) -> Optional[Dict]:
        """
        查询 TopK baseline

        Args:
            hardware: 硬件名称
            k: Top-K 的 K 值
            ne0: 第一个维度大小
            ne: 张量形状 [ne0, ne1, ne2, ne3]

        Returns:
            {'gbps': float, 'us_per_run': float}
        """
        ne_str = "x".join(map(str, ne))
        case_id = f"topk_k{k}_ne0{ne0}_{ne_str}"
        data = self._data.get("topk", {}).get(case_id, {})
        hw_data = data.get("hardware", {}).get(hardware)

        if hw_data:
            return {
                "gbps": hw_data.get("gbps"),
                "us_per_run": hw_data.get("us_per_run")
            }
        return None

    def get_topk_k_values(self) -> List[int]:
        """获取所有支持的 TopK k 值"""
        k_values = set()
        for case_data in self._data.get("topk", {}).values():
            k = case_data.get("k")
            if k:
                k_values.add(k)
        return sorted(list(k_values))

    # ==================== 通用方法 ====================

    def get(
        self,
        op_type: str,
        hardware: str,
        **kwargs
    ) -> Optional[Dict]:
        """
        通用查询接口

        Args:
            op_type: 算子类型 (gemm, flash_attention, rms_norm, topk)
            hardware: 硬件名称
            **kwargs: 算子特定参数

        Returns:
            baseline 数据
        """
        op_type = op_type.lower()

        if op_type in ("gemm", "quant_gemm"):
            return self.get_gemm(
                hardware,
                kwargs.get("type_a", "q4_0"),
                kwargs.get("m", 4096),
                kwargs.get("n", 1),
                kwargs.get("k", 4096),
                kwargs.get("type_b", "f32")
            )
        elif op_type == "flash_attention":
            return self.get_flash_attn(
                hardware,
                kwargs.get("model", "Llama3-8B"),
                kwargs.get("kv_type", "F16"),
                kwargs.get("cache_size", 512),
                kwargs.get("batch_size", 512)
            )
        elif op_type == "rms_norm":
            return self.get_rms_norm(
                hardware,
                kwargs.get("hidden_size", 4096),
                kwargs.get("ne", [4096, 512, 1, 1])
            )
        elif op_type == "topk":
            return self.get_topk(
                hardware,
                kwargs.get("k", 8),
                kwargs.get("ne0", 256),
                kwargs.get("ne", [256, 512, 1, 1])
            )
        else:
            raise ValueError(f"Unknown operator type: {op_type}")

    def get_metric_type(self, op_type: str) -> str:
        """获取算子的性能指标类型"""
        return self.OPERATOR_METRICS.get(op_type.lower(), "unknown")

    def get_supported_hardware(self, op_type: str = None) -> List[str]:
        """获取支持的硬件列表"""
        hardware = set()

        data_sources = self._data.values() if op_type is None else [self._data.get(op_type, {})]

        for data in data_sources:
            for case_data in data.values():
                if isinstance(case_data, dict) and "hardware" in case_data:
                    hardware.update(case_data["hardware"].keys())

        return sorted(list(hardware))

    def summary(self) -> Dict:
        """获取 baseline 数据摘要"""
        return {
            "gemm": {
                "count": len(self._data.get("gemm", {})),
                "hardware": self.get_supported_hardware("gemm")
            },
            "flash_attention": {
                "count": len(self._data.get("flash_attention", {})),
                "models": self.get_flash_attn_models(),
                "hardware": self.get_supported_hardware("flash_attention")
            },
            "rms_norm": {
                "count": len(self._data.get("rms_norm", {})),
                "hidden_sizes": self.get_rms_norm_hidden_sizes(),
                "hardware": self.get_supported_hardware("rms_norm")
            },
            "topk": {
                "count": len(self._data.get("topk", {})),
                "k_values": self.get_topk_k_values(),
                "hardware": self.get_supported_hardware("topk")
            }
        }

    def print_summary(self):
        """打印 baseline 数据摘要"""
        print("\n" + "=" * 60)
        print("Baseline API 数据摘要")
        print("=" * 60)

        summary = self.summary()

        for op_type, info in summary.items():
            print(f"\n【{op_type.upper()}】")
            print(f"  数据量: {info['count']} entries")
            print(f"  硬件: {info['hardware']}")

            if "models" in info:
                print(f"  模型: {info['models']}")
            if "hidden_sizes" in info:
                print(f"  Hidden sizes: {info['hidden_sizes']}")
            if "k_values" in info:
                print(f"  K values: {info['k_values']}")

        print("\n" + "=" * 60)


# 单例缓存，避免每次 query_baseline 重复加载 ~600KB JSON
_cached_api: Optional[BaselineAPI] = None


def get_baseline_api(baseline_dir: str = None) -> BaselineAPI:
    """获取单例 BaselineAPI 实例，避免批量测试时重复加载数据文件。"""
    global _cached_api
    if _cached_api is None:
        _cached_api = BaselineAPI(baseline_dir=baseline_dir)
    return _cached_api


# 便捷函数
def get_baseline(op_type: str, hardware: str, **kwargs) -> Optional[Dict]:
    """快速查询 baseline"""
    return get_baseline_api().get(op_type, hardware, **kwargs)


# 向后兼容的别名
GEMMBaselineAPI = BaselineAPI
NewOperatorsBaselineAPI = BaselineAPI


if __name__ == "__main__":
    api = get_baseline_api()
    api.print_summary()

    # 测试查询
    print("\n" + "=" * 60)
    print("查询示例")
    print("=" * 60)

    # GEMM
    result = api.get_gemm("RTX4090", "q4_0", 4096, 1, 4096)
    print(f"\nGEMM: {result}")

    # Flash Attention
    result = api.get_flash_attn("RTX4090", "Llama3-8B", "F16", 512, 512)
    print(f"Flash Attention: {result}")

    # RMS Norm
    result = api.get_rms_norm("RTX4090", 4096, [4096, 512, 1, 1])
    print(f"RMS Norm: {result}")

    # TopK
    result = api.get_topk("RTX4090", 8, 256, [256, 512, 1, 1])
    print(f"TopK: {result}")

    # 通用接口
    result = api.get("flash_attention", "RTX4090", model="Llama3-8B", kv_type="F16", cache_size=512, batch_size=512)
    print(f"通用接口: {result}")
