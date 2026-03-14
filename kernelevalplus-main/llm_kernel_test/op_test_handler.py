#!/usr/bin/env python3
"""
统一算子测试框架

提供可扩展的 Handler 模式，支持多种算子类型：
- Quantized GEMM (q4_0, q4_1, q8_0, q8_1)
- Flash Attention
- RMS Norm
- TopK

使用方法:
    # 统一测试命令
    python llm_kernel_test/unified_test_runner.py --test \\
        --definition definitions/<op_type>/<model>/<name>.json \\
        --attempt-path attempts/<name>_v<N>

添加新算子:
    1. 继承 OperatorTestHandler 类
    2. 实现所有抽象方法
    3. 注册到 OPERATOR_HANDLERS 字典
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import re
import torch


class OperatorTestHandler(ABC):
    """
    算子测试处理器基类

    所有算子必须实现核心方法，可选实现高级功能方法。
    """

    @property
    @abstractmethod
    def op_type(self) -> str:
        """算子类型标识符"""
        pass

    @property
    @abstractmethod
    def performance_metric(self) -> str:
        """性能指标: 'tflops' 或 'gbps'"""
        pass

    # ==================== 核心方法 (必须实现) ====================

    @abstractmethod
    def generate_inputs(self, spec: Dict, test_config: Dict, device: str) -> Dict[str, torch.Tensor]:
        """生成测试输入数据"""
        pass

    @abstractmethod
    def get_reference_output(self, spec: Dict, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用参考实现计算预期输出"""
        pass

    @abstractmethod
    def run_kernel(self, kernel_func, inputs: Dict[str, torch.Tensor], spec: Dict) -> torch.Tensor:
        """运行 kernel 函数"""
        pass

    @abstractmethod
    def calculate_performance(self, output: torch.Tensor, latency_ms: float, spec: Dict, test_config: Dict) -> Dict:
        """计算性能指标 (TFLOPS 或 GB/s)"""
        pass

    @abstractmethod
    def query_baseline(self, spec: Dict, hardware: str, test_config: Dict) -> Optional[Dict]:
        """查询 baseline 性能数据"""
        pass

    # ==================== 高级功能 (可选实现) ====================

    def generate_wrapper(self, spec: Dict, kernel_code: str) -> Optional[str]:
        """
        生成 PyTorch wrapper 代码

        当 kernel 使用 extern "C" 风格时，需要生成适配层。
        返回 None 表示不需要 wrapper。
        """
        return None

    def diagnose_compilation_error(self, error_msg: str, kernel_code: str) -> List[Dict]:
        """
        诊断编译错误，返回修复建议

        Returns:
            List of {"type": str, "problem": str, "suggestion": str}
        """
        return []

    def preprocess_kernel(self, kernel_code: str) -> str:
        """
        预处理 kernel 代码

        可用于自动修复常见问题（如 block size 超限）。
        返回修改后的代码。
        """
        return kernel_code

    def get_test_configs(self, spec: Dict) -> List[Dict]:
        """
        获取测试配置列表

        Returns:
            List of test config dicts, e.g. [{"M": 1, "name": "single_token"}, ...]
        """
        return spec.get("test_configs", [{"name": "default"}])

    def validate_output_shape(self, output: torch.Tensor, ref_output: torch.Tensor) -> torch.Tensor:
        """
        验证输出形状，处理转置等问题

        Returns:
            可能调整后的 output
        """
        if output.shape == ref_output.shape:
            return output
        if output.shape == ref_output.T.shape:
            return output.T
        if output.numel() == ref_output.numel():
            return output.reshape(ref_output.shape)
        return output


class QuantGEMMHandler(OperatorTestHandler):
    """
    Quantized GEMM 测试处理器 - 完整实现

    支持量化格式: q4_0, q4_1, q8_0, q8_1
    包含完整的编译错误诊断和代码预处理功能
    """

    # 量化格式配置
    BLOCK_SIZES = {
        "block_q4_0": 18,
        "block_q4_1": 20,
        "block_q5_0": 22,
        "block_q5_1": 24,
        "block_q8_0": 34,
        "block_q8_1": 36,
    }

    @property
    def op_type(self) -> str:
        return "quant_gemm"

    @property
    def performance_metric(self) -> str:
        return "tflops"

    def _get_dims(self, spec: Dict, test_config: Dict) -> tuple:
        """从 spec 或 test_config 获取维度参数"""
        axes = spec.get("axes", {})

        def get_dim(name, default=4096):
            # 优先从 test_config 获取
            if name in test_config:
                return test_config[name]
            # 其次从 axes 获取
            axis_info = axes.get(name, {})
            if isinstance(axis_info, dict):
                if axis_info.get("type") == "const":
                    return axis_info.get("value", default)
                return axis_info.get("default", default)
            return default

        M = get_dim("M", 1)
        N = get_dim("N", 4096)
        K = get_dim("K", 4096)
        return M, N, K

    def generate_inputs(self, spec: Dict, test_config: Dict, device: str) -> Dict[str, torch.Tensor]:
        """生成量化权重和激活"""
        from llm_kernel_test.reference import (
            quantize_to_q4_0, quantize_to_q4_1,
            quantize_to_q8_0, quantize_to_q8_1
        )

        M, N, K = self._get_dims(spec, test_config)

        weight_dtype = spec.get("inputs", {}).get("weight", {}).get("dtype", "block_q8_0")
        activation_dtype = spec.get("inputs", {}).get("activation", {}).get("dtype", "float32")

        # 生成 FP32 权重并量化
        weight_fp32 = torch.randn(N, K, dtype=torch.float32)

        # Support both "q4_0" and "block_q4_0" formats
        if weight_dtype in ("q4_0", "block_q4_0"):
            weight_q_bytes = quantize_to_q4_0(weight_fp32)
        elif weight_dtype in ("q4_1", "block_q4_1"):
            weight_q_bytes = quantize_to_q4_1(weight_fp32)
        elif weight_dtype in ("q8_0", "block_q8_0"):
            weight_q_bytes = quantize_to_q8_0(weight_fp32)
        elif weight_dtype in ("q8_1", "block_q8_1"):
            weight_q_bytes = quantize_to_q8_1(weight_fp32)
        else:
            # 默认 Q8_0
            weight_q_bytes = quantize_to_q8_0(weight_fp32)

        weight_q = torch.from_numpy(weight_q_bytes).to(device)

        # 生成激活
        activation_fp32 = torch.randn(M, K, dtype=torch.float32)
        is_fp32_activation = activation_dtype in ("float32", "float16", "bfloat16", "fp32")

        if is_fp32_activation:
            activation = activation_fp32.to(device)
        else:
            # 预量化激活
            if activation_dtype in ("q8_1", "block_q8_1"):
                act_q_bytes = quantize_to_q8_1(activation_fp32)
            else:
                act_q_bytes = quantize_to_q8_0(activation_fp32)
            activation = torch.from_numpy(act_q_bytes).to(device)

        return {
            "weight": weight_q,
            "activation": activation,
            "M": M, "N": N, "K": K,
            "weight_dtype": weight_dtype,
            "activation_dtype": activation_dtype
        }

    def get_reference_output(self, spec: Dict, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用参考实现计算输出"""
        from llm_kernel_test.reference import (
            reference_q4_0_fp32_gemm, reference_q4_1_fp32_gemm,
            reference_q8_0_gemm, reference_q4_0_q8_1_gemm
        )

        weight = inputs["weight"]
        activation = inputs["activation"]
        M, N, K = inputs["M"], inputs["N"], inputs["K"]
        weight_dtype = inputs.get("weight_dtype", "block_q8_0")
        activation_dtype = inputs.get("activation_dtype", "float32")
        is_fp32_activation = activation_dtype in ("float32", "float16", "bfloat16", "fp32")

        if weight_dtype in ("q4_0", "block_q4_0") and is_fp32_activation:
            return reference_q4_0_fp32_gemm(weight, activation, M, N, K)
        elif weight_dtype in ("q4_1", "block_q4_1") and is_fp32_activation:
            return reference_q4_1_fp32_gemm(weight, activation, M, N, K)
        elif weight_dtype in ("q4_0", "block_q4_0") and activation_dtype in ("q8_1", "block_q8_1"):
            return reference_q4_0_q8_1_gemm(weight, activation, M, N, K)
        else:
            # 默认 Q8_0 + FP32
            return reference_q8_0_gemm(weight, activation, M, N, K)

    def run_kernel(self, kernel_func, inputs: Dict[str, torch.Tensor], spec: Dict) -> torch.Tensor:
        """运行 kernel: forward(weight, activation, M, N, K)"""
        return kernel_func(
            inputs["weight"], inputs["activation"],
            inputs["M"], inputs["N"], inputs["K"]
        )

    def calculate_performance(self, output: torch.Tensor, latency_ms: float, spec: Dict, test_config: Dict) -> Dict:
        """计算 TFLOPS"""
        M, N, K = self._get_dims(spec, test_config)

        total_ops = 2 * M * N * K  # multiply-add
        flops = total_ops / (latency_ms / 1000)
        tflops = flops / 1e12

        return {"tflops": round(tflops, 3), "gflops": round(flops / 1e9, 1)}

    def query_baseline(self, spec: Dict, hardware: str, test_config: Dict) -> Optional[Dict]:
        """查询 GGML baseline"""
        try:
            from core.tools.baseline_api import get_baseline_api
            api = get_baseline_api()

            M, N, K = self._get_dims(spec, test_config)

            # 维度映射: 我们的 M=批大小, GGML 的 N=批大小
            weight_dtype = spec.get("inputs", {}).get("weight", {}).get("dtype", "block_q8_0")
            type_a = weight_dtype.replace("block_", "")

            # GGML 维度: M = N (输出特征), N = M (批大小), K = K
            baseline = api.get_baseline(hardware, type_a, N, M, K)

            if baseline:
                return {
                    "hardware": hardware,
                    "type_a": type_a,
                    "dimensions": {"M": M, "N": N, "K": K},
                    "gflops": baseline.get("gflops"),
                    "tflops": baseline.get("tflops"),
                }
        except Exception as e:
            print(f"  ⚠️ Baseline 查询失败: {e}")

        return None

    # ==================== 高级功能 ====================

    def generate_wrapper(self, spec: Dict, kernel_code: str) -> Optional[str]:
        """生成 GEMM PyTorch wrapper"""
        if 'PYBIND11_MODULE' in kernel_code:
            return None

        # 检测 extern "C" 函数
        extern_match = re.search(r'extern\s+"C"\s+void\s+(\w+)\s*\(', kernel_code)
        if not extern_match:
            return None

        entry_point = extern_match.group(1)

        # 确定 activation 类型
        activation_dtype = spec.get("inputs", {}).get("activation", {}).get("dtype", "float32")
        is_fp32 = activation_dtype in ("float32", "float16", "bfloat16", "fp32")
        act_cpp_type = "float" if is_fp32 else "uint8_t"

        wrapper = f'''#include <torch/extension.h>
#include <cstdint>

extern "C" void {entry_point}(
    const uint8_t* weight,
    const {act_cpp_type}* activation,
    float* output,
    int M, int N, int K
);

torch::Tensor forward(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t M, int64_t N, int64_t K
) {{
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be CUDA tensor");

    auto output = torch::empty({{M, N}}, torch::dtype(torch::kFloat32).device(weight.device()));

    {entry_point}(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<{act_cpp_type}>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &forward, "Quantized GEMM");
}}
'''
        return wrapper

    def diagnose_compilation_error(self, error_msg: str, kernel_code: str) -> List[Dict]:
        """GEMM 专属的编译错误诊断"""
        diagnostics = []
        error_lower = error_msg.lower()

        # 1. FP16 转换错误
        if ('__half2float' in error_msg or
            ('no suitable constructor' in error_lower and '__half' in error_msg)):
            diagnostics.append({
                "type": "fp16_conversion",
                "problem": "FP16 转换错误: 直接使用 __half2float(uint16_t) 失败",
                "suggestion": "使用 union 方式转换:\n"
                             "     __device__ float read_half_as_float(uint16_t h) {\n"
                             "         union { uint16_t u16; __half f16; } un;\n"
                             "         un.u16 = h;\n"
                             "         return __half2float(un.f16);\n"
                             "     }\n"
                             "     然后使用: float d_w = read_half_as_float(w_block->d);"
            })

        # 2. 量化结构体未定义
        struct_defs = {
            'block_q4_0': "typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0; static_assert(sizeof(block_q4_0) == 18, \"\");",
            'block_q8_0': "typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0; static_assert(sizeof(block_q8_0) == 34, \"\");",
            'block_q8_1': "typedef struct { uint32_t ds; int8_t qs[32]; } block_q8_1; static_assert(sizeof(block_q8_1) == 36, \"\");",
        }

        missing_structs = []
        for struct_name in struct_defs:
            if struct_name in error_msg and ('undefined' in error_lower or 'undeclared' in error_lower):
                missing_structs.append(struct_name)

        if missing_structs:
            suggestion = "添加以下结构体定义:\n"
            for s in missing_structs:
                suggestion += f"     {struct_defs[s]}\n"
            diagnostics.append({
                "type": "missing_struct",
                "problem": f"结构体未定义: {', '.join(missing_structs)}",
                "suggestion": suggestion
            })

        # 3. 缺少头文件
        missing_headers = []
        if 'cuda_runtime.h' not in kernel_code and 'cudaError' in error_msg:
            missing_headers.append('cuda_runtime.h')
        if 'cuda_fp16.h' not in kernel_code and '__half' in error_msg:
            missing_headers.append('cuda_fp16.h')
        if 'stdint.h' not in kernel_code and ('uint16_t' in error_msg or 'int8_t' in error_msg):
            missing_headers.append('stdint.h')

        if missing_headers:
            diagnostics.append({
                "type": "missing_headers",
                "problem": f"缺少必要的头文件: {', '.join(missing_headers)}",
                "suggestion": "添加: " + " ".join(f"#include <{h}>" for h in missing_headers)
            })

        # 4. 缺少 PYBIND11_MODULE
        if ('pybind11' in error_lower or 'torch_extension' in error_lower):
            if 'PYBIND11_MODULE' not in kernel_code:
                diagnostics.append({
                    "type": "missing_pybind",
                    "problem": "缺少 PyTorch 绑定",
                    "suggestion": "在 kernel 末尾添加:\n"
                                 "     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
                                 "         m.def(\"forward\", &forward);\n"
                                 "     }"
                })

        return diagnostics

    def preprocess_kernel(self, kernel_code: str) -> str:
        """自动修复 block size 超过 1024 的问题"""
        def fix_block_size(match):
            x = int(match.group(1))
            y = int(match.group(2)) if match.group(2) else 1
            total = x * y

            if total > 1024:
                # 计算安全配置
                if x >= y:
                    new_x = min(x, 1024)
                    new_y = max(1, 1024 // new_x)
                else:
                    new_y = min(y, 1024)
                    new_x = max(1, 1024 // new_y)
                return f'dim3 block({new_x}, {new_y})'
            return match.group(0)

        # 匹配 dim3 block(x, y) 或 dim3 block(x)
        kernel_code = re.sub(
            r'dim3\s+block\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)',
            fix_block_size, kernel_code
        )

        return kernel_code

    def get_test_configs(self, spec: Dict) -> List[Dict]:
        """获取 GEMM 测试配置"""
        M, N, K = self._get_dims(spec, {})

        # 默认测试配置: 单 token, 小批量, 大批量
        return [
            {"M": 1, "N": N, "K": K, "name": "single_token"},
            {"M": 8, "N": N, "K": K, "name": "small_batch"},
            {"M": 512, "N": N, "K": K, "name": "large_batch"},
        ]


class FlashAttentionHandler(OperatorTestHandler):
    """Flash Attention 测试处理器"""

    @property
    def op_type(self) -> str:
        return "flash_attention"

    @property
    def performance_metric(self) -> str:
        return "tflops"

    def generate_inputs(self, spec: Dict, test_config: Dict, device: str) -> Dict[str, torch.Tensor]:
        """生成 Q, K, V 缓存"""
        batch_size = test_config.get("batch_size", 1)
        seq_len = spec["axes"]["seq_len"]["value"]
        num_heads = spec["axes"]["num_heads"]["value"]
        head_dim = spec["axes"]["head_dim"]["value"]

        # Query: [batch_size, num_heads, head_dim]
        query = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float32, device=device)

        # Determine KV dtype from spec
        kv_dtype_str = spec.get("inputs", {}).get("key_cache", {}).get("dtype", "float16")

        # Generate FP32 KV cache first (for reference computation)
        key_cache_fp32 = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        value_cache_fp32 = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32, device=device)

        if kv_dtype_str == "int8":
            # Determine quantization format from variant or tags
            variant = spec.get("variant", "").lower()
            tags = spec.get("tags", [])
            is_q8 = "q8" in variant or any("q8" in tag for tag in tags)

            if is_q8:
                # Q8_0 format
                from llm_kernel_test.reference import quantize_to_q8_0

                # Reshape to [seq_len * num_heads, head_dim] for quantization
                key_flat = key_cache_fp32.reshape(-1, head_dim)  # [S*H, D]
                value_flat = value_cache_fp32.reshape(-1, head_dim)  # [S*H, D]

                # Quantize to Q8_0
                key_q = quantize_to_q8_0(key_flat.cpu().numpy())
                value_q = quantize_to_q8_0(value_flat.cpu().numpy())

                key_cache = torch.from_numpy(key_q).to(device)
                value_cache = torch.from_numpy(value_q).to(device)
            else:
                # Q4_0 format
                from llm_kernel_test.reference import quantize_to_q4_0

                # Reshape to [seq_len * num_heads, head_dim] for quantization
                key_flat = key_cache_fp32.reshape(-1, head_dim)  # [S*H, D]
                value_flat = value_cache_fp32.reshape(-1, head_dim)  # [S*H, D]

                # Quantize to Q4_0
                key_q = quantize_to_q4_0(key_flat.cpu().numpy())
                value_q = quantize_to_q4_0(value_flat.cpu().numpy())

                key_cache = torch.from_numpy(key_q).to(device)
                value_cache = torch.from_numpy(value_q).to(device)
        elif kv_dtype_str == "float16":
            key_cache = key_cache_fp32.half()
            value_cache = value_cache_fp32.half()
        elif kv_dtype_str == "float32":
            key_cache = key_cache_fp32
            value_cache = value_cache_fp32
        else:
            # Default to FP16
            key_cache = key_cache_fp32.half()
            value_cache = value_cache_fp32.half()

        return {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "key_cache_fp32": key_cache_fp32,  # Keep FP32 for reference
            "value_cache_fp32": value_cache_fp32,  # Keep FP32 for reference
            "kv_dtype_str": kv_dtype_str
        }

    def get_reference_output(self, spec: Dict, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用 PyTorch 参考 Attention 计算"""
        query = inputs["query"].float()  # [B, H, D]

        # Use FP32 KV cache for reference computation (accounts for quantization error)
        if "key_cache_fp32" in inputs:
            key = inputs["key_cache_fp32"].float()  # [S, H, D]
            value = inputs["value_cache_fp32"].float()  # [S, H, D]
        else:
            key = inputs["key_cache"].float()  # [S, H, D]
            value = inputs["value_cache"].float()  # [S, H, D]

        # 简化版 Attention: Q @ K^T @ V
        # Q: [B, H, D], K: [S, H, D] -> attn: [B, H, S]
        attn_weights = torch.einsum('bhd,shd->bhs', query, key) / (query.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # attn: [B, H, S], V: [S, H, D] -> output: [B, H, D]
        output = torch.einsum('bhs,shd->bhd', attn_weights, value)

        return output

    def run_kernel(self, kernel_func, inputs: Dict[str, torch.Tensor], spec: Dict) -> torch.Tensor:
        """运行 Flash Attention kernel"""
        # Kernel 接口: forward(query, key_cache, value_cache, batch_size, seq_len, num_heads, head_dim)
        batch_size = inputs["query"].shape[0]
        # Get seq_len from spec, not from key_cache shape (which is different for quantized formats)
        seq_len = spec["axes"]["seq_len"]["value"]
        num_heads = inputs["query"].shape[1]
        head_dim = inputs["query"].shape[2]

        return kernel_func(
            inputs["query"],
            inputs["key_cache"],
            inputs["value_cache"],
            batch_size, seq_len, num_heads, head_dim
        )

    def calculate_performance(self, output: torch.Tensor, latency_ms: float, spec: Dict, test_config: Dict) -> Dict:
        """计算 Attention TFLOPS"""
        batch_size = test_config.get("batch_size", 1)
        seq_len = spec["axes"]["seq_len"]["value"]
        num_heads = spec["axes"]["num_heads"]["value"]
        head_dim = spec["axes"]["head_dim"]["value"]

        # Attention FLOPs: Q@K^T + softmax + attn@V
        # Q@K^T: B * H * S * D * 2
        # attn@V: B * H * S * D * 2
        flops_per_attention = 4 * batch_size * num_heads * seq_len * head_dim
        total_ops = flops_per_attention

        flops = total_ops / (latency_ms / 1000)
        tflops = flops / 1e12

        return {"tflops": round(tflops, 3)}

    def query_baseline(self, spec: Dict, hardware: str, test_config: Dict) -> Optional[Dict]:
        """查询 Flash Attention baseline"""
        try:
            from core.tools.baseline_api import get_baseline_api
            api = get_baseline_api()

            # 从 spec 提取参数
            model = spec.get("model_architectures", ["unknown"])[0]
            if "llama" in model.lower():
                model = "Llama3-8B"
            elif "qwen" in model.lower():
                model = "Qwen2.5-7B"

            kv_type = "F16"
            for tag in spec.get("tags", []):
                if "kv_cache:" in tag:
                    kv_type_raw = tag.split(":")[1].upper()
                    if kv_type_raw in ("F16", "Q4_0", "Q8_0"):
                        kv_type = kv_type_raw
                    break

            cache_size = spec["axes"]["seq_len"]["value"]
            batch_size = test_config.get("batch_size", 512)

            baseline = api.get_flash_attn(hardware, model, kv_type, cache_size, batch_size)

            if baseline:
                return {
                    "model": model,
                    "kv_type": kv_type,
                    "cache_size": cache_size,
                    "batch_size": batch_size,
                    "tflops": baseline.get("tflops"),
                    "us_per_run": baseline.get("us_per_run")
                }

        except Exception as e:
            print(f"  ⚠️ Baseline 查询失败: {e}")

        return None


class RMSNormHandler(OperatorTestHandler):
    """RMS Norm 测试处理器"""

    @property
    def op_type(self) -> str:
        return "rms_norm"

    @property
    def performance_metric(self) -> str:
        return "gbps"  # 内存带宽敏感

    def generate_inputs(self, spec: Dict, test_config: Dict, device: str) -> Dict[str, torch.Tensor]:
        """生成输入和权重"""
        batch_size = test_config.get("batch_size", 1)
        hidden_size = spec["axes"]["hidden_size"]["value"]

        hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.float32, device=device)
        weight = torch.randn(hidden_size, dtype=torch.float32, device=device)

        return {"hidden_states": hidden_states, "weight": weight}

    def get_reference_output(self, spec: Dict, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """RMS Norm 参考实现"""
        hidden_states = inputs["hidden_states"]
        weight = inputs["weight"]
        epsilon = spec["axes"].get("epsilon", {}).get("value", 1e-6)

        # RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(hidden_states ** 2, dim=-1, keepdim=True) + epsilon)
        output = hidden_states / rms * weight

        return output

    def run_kernel(self, kernel_func, inputs: Dict[str, torch.Tensor], spec: Dict) -> torch.Tensor:
        """运行 RMS Norm kernel"""
        return kernel_func(inputs["hidden_states"], inputs["weight"])

    def calculate_performance(self, output: torch.Tensor, latency_ms: float, spec: Dict, test_config: Dict) -> Dict:
        """计算 GB/s"""
        batch_size = test_config.get("batch_size", 1)
        hidden_size = spec["axes"]["hidden_size"]["value"]

        # 数据量: input (FP32) + weight (FP32) + output (FP32)
        bytes_per_element = 4  # FP32
        total_bytes = (batch_size * hidden_size * 2 + hidden_size + batch_size * hidden_size) * bytes_per_element

        bandwidth = total_bytes / (latency_ms / 1000) / 1e9  # GB/s

        return {"gbps": round(bandwidth, 2)}

    def query_baseline(self, spec: Dict, hardware: str, test_config: Dict) -> Optional[Dict]:
        """查询 RMS Norm baseline"""
        try:
            from core.tools.baseline_api import get_baseline_api
            api = get_baseline_api()

            hidden_size = spec["axes"]["hidden_size"]["value"]
            batch_size = test_config.get("batch_size", 512)

            # 构建 ne: [hidden_size, 1, batch_size, 1]
            ne = [hidden_size, 1, batch_size, 1]

            baseline = api.get_rms_norm(hardware, hidden_size, ne)

            if baseline:
                return {
                    "hidden_size": hidden_size,
                    "batch_size": batch_size,
                    "gbps": baseline.get("gbps"),
                    "us_per_run": baseline.get("us_per_run")
                }

        except Exception as e:
            print(f"  ⚠️ Baseline 查询失败: {e}")

        return None


class TopKHandler(OperatorTestHandler):
    """TopK 测试处理器"""

    @property
    def op_type(self) -> str:
        return "topk"

    @property
    def performance_metric(self) -> str:
        return "gbps"

    def generate_inputs(self, spec: Dict, test_config: Dict, device: str) -> Dict[str, torch.Tensor]:
        """生成概率分布"""
        batch_size = test_config.get("batch_size", 1)
        vocab_size = spec["axes"].get("vocab_size", {}).get("value", 256)
        vocab_subset = spec["axes"].get("vocab_subset", {}).get("value", vocab_size)

        # 生成随机概率分布
        probs = torch.rand(batch_size, vocab_subset, dtype=torch.float32, device=device)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 归一化

        return {"probs": probs}

    def get_reference_output(self, spec: Dict, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        TopK 参考实现 - 返回从 top-k 中采样的结果

        根据公式: top_indices = argsort(probs, descending=True)[:k]; samples = categorical(probs[top_indices])

        对于测试目的，使用确定性采样（argmax）以确保可复现性
        """
        probs = inputs["probs"]
        k = spec["axes"]["k"]["value"]

        # 1. 获取 top-k 值和索引
        topk_values, topk_indices = torch.topk(probs, k, dim=-1)

        # 2. 确定性采样: 选择 top-1 (argmax)
        # 这确保了与 CUDA kernel 的可复现性匹配
        # 对于实际推理，可以使用 stochastic sampling
        sampled_indices = topk_indices[:, 0]  # 选择概率最大的索引

        return sampled_indices  # 返回 [batch_size]，每个批次一个采样结果

    def run_kernel(self, kernel_func, inputs: Dict[str, torch.Tensor], spec: Dict) -> torch.Tensor:
        """运行 TopK kernel"""
        k = spec["axes"]["k"]["value"]
        return kernel_func(inputs["probs"], k)

    def calculate_performance(self, output: torch.Tensor, latency_ms: float, spec: Dict, test_config: Dict) -> Dict:
        """计算 GB/s"""
        batch_size = test_config.get("batch_size", 1)
        vocab_size = spec["axes"].get("vocab_subset", spec["axes"].get("vocab_size", {})).get("value", 256)

        # 数据量: input probs (FP32)
        bytes_per_element = 4
        total_bytes = batch_size * vocab_size * bytes_per_element

        bandwidth = total_bytes / (latency_ms / 1000) / 1e9

        return {"gbps": round(bandwidth, 2)}

    def query_baseline(self, spec: Dict, hardware: str, test_config: Dict) -> Optional[Dict]:
        """查询 TopK baseline"""
        try:
            from core.tools.baseline_api import get_baseline_api
            api = get_baseline_api()

            k = spec["axes"]["k"]["value"]
            # TopK baseline 使用 vocab_subset=256
            vocab_subset = 256
            batch_size = test_config.get("batch_size", 512)

            # 构建查询的 ne
            ne = [vocab_subset, batch_size, 1, 1]

            baseline = api.get_topk(hardware, k, vocab_subset, ne)

            if baseline:
                return {
                    "k": k,
                    "vocab_subset": vocab_subset,
                    "batch_size": batch_size,
                    "gbps": baseline.get("gbps"),
                    "us_per_run": baseline.get("us_per_run")
                }

        except Exception as e:
            print(f"  ⚠️ Baseline 查询失败: {e}")

        return None


# 算子处理器注册表
OPERATOR_HANDLERS = {
    "quant_gemm": QuantGEMMHandler(),
    "flash_attention": FlashAttentionHandler(),
    "rms_norm": RMSNormHandler(),
    "topk": TopKHandler(),
}


def get_handler(op_type: str) -> Optional[OperatorTestHandler]:
    """获取算子处理器"""
    return OPERATOR_HANDLERS.get(op_type)


def detect_op_type(spec: Dict) -> str:
    """从 spec 中检测算子类型"""
    # 优先使用显式声明
    if "op_type" in spec:
        return spec["op_type"]

    # 从 inputs 推断
    inputs = spec.get("inputs", {})

    if "query" in inputs and "key_cache" in inputs:
        return "flash_attention"
    elif "hidden_states" in inputs and "weight" in inputs and "output" in spec.get("outputs", {}):
        # 可能是 norm 层
        if "rms" in str(spec.get("tags", [])).lower():
            return "rms_norm"
    elif "probs" in inputs:
        return "topk"
    elif "weight" in inputs and "activation" in inputs:
        return "quant_gemm"

    return "quant_gemm"  # 默认
