"""Prompt templates for KernelEvalPlus task."""

from __future__ import annotations

import json
from typing import Any


def _kernel_entry_signature_text(op_type: str) -> str:
    op = str(op_type or "").strip()
    if op == "flash_attention":
        return (
            "extern \"C\" void flash_attn_kernel(\n"
            "    const float* query,\n"
            "    const void* key_cache,\n"
            "    const void* value_cache,\n"
            "    float* output,\n"
            "    int batch_size, int seq_len, int num_heads, int head_dim\n"
            ");"
        )
    if op == "rms_norm":
        return (
            "extern \"C\" void rms_norm_kernel(\n"
            "    const float* input,\n"
            "    const float* weight,\n"
            "    float* output,\n"
            "    int batch_size, int hidden_size\n"
            ");"
        )
    if op == "topk":
        return (
            "extern \"C\" void topk_kernel(\n"
            "    const float* probs,\n"
            "    int64_t* indices,\n"
            "    int batch_size, int vocab_size, int k\n"
            ");"
        )
    if op == "quant_gemm":
        return (
            "extern \"C\" void <entry_point>(\n"
            "    const uint8_t* weight,\n"
            "    const <activation_type>* activation,\n"
            "    float* output,\n"
            "    int M, int N, int K\n"
            ");\n"
            "Note: <activation_type> is float for fp32/fp16/bf16 activation, "
            "otherwise uint8_t for quantized activations."
        )
    return "extern \"C\" void <kernel_name>(...);"


def _op_specific_guidance(op_type: str) -> str:
    op = str(op_type or "").strip()
    if op == "flash_attention":
        return (
            "FlashAttention guidance:\n"
            "- Implement `flash_attn_kernel` and launch a CUDA kernel inside it.\n"
            "- query is float*; key/value are void* (quant or fp); output is float*.\n"
            "- Focus on memory coalescing and minimize global memory passes."
        )
    if op == "rms_norm":
        return (
            "RMSNorm guidance:\n"
            "- Implement `rms_norm_kernel` and launch CUDA kernel inside it.\n"
            "- Use float accumulation for stability; avoid extra global writes.\n"
            "- Prefer vectorized loads/stores when hidden_size is divisible."
        )
    if op == "topk":
        return (
            "TopK guidance:\n"
            "- Implement `topk_kernel` to compute top-k indices per row.\n"
            "- Avoid full sort when k is small; use partial selection.\n"
            "- Ensure deterministic behavior for equal values if possible."
        )
    if op == "quant_gemm":
        return (
            "QuantGEMM guidance:\n"
            "- Define one extern \"C\" entry point with the required signature.\n"
            "- The wrapper will call your entry point directly.\n"
            "- Use shared memory to stage blocks; consider DP4A or tensor core paths if applicable."
        )
    return "General guidance:\n- Implement the extern \"C\" entry and launch a CUDA kernel inside it."


def get_definition_text(*, definition: dict[str, Any], op_type: str, language: str | None = None) -> str:
    lang = str(language or "").strip().lower()
    if lang and lang != "cuda":
        raise ValueError("KernelEvalPlus prompts support only CUDA language.")
    spec_json = json.dumps(definition, ensure_ascii=False, indent=2)
    return (
        "You are writing a CUDA kernel implementation for KernelEvalPlus.\n"
        "Provide a single file named kernel.cu. Do not include PYBIND11_MODULE; "
        "the test runner will generate wrappers if needed.\n\n"
        "Required extern \"C\" entry signature:\n"
        f"{_kernel_entry_signature_text(op_type)}\n\n"
        f"{_op_specific_guidance(op_type)}\n\n"
        "Operator definition (JSON):\n"
        f"{spec_json}\n"
    )


def get_code_format_text(*, language: str, target_gpu: str) -> str:
    lang = str(language or "").strip().lower()
    if lang != "cuda":
        return ""
    return (
        "Output format: XML blocks with EXACT tags:\n"
        "<header_file name=\"kernel.h\">...optional...</header_file>\n"
        "<cuda_file name=\"kernel.cu\">...your kernel code...</cuda_file>\n"
        "<cpp_file name=\"main.cpp\">...optional stub...</cpp_file>\n"
        "The evaluator reads kernel.cu only; kernel.h/main.cpp can be minimal."
    )


def get_generation_prompt(*, definition: dict[str, Any], op_type: str, language: str, target_gpu: str) -> str:
    base = get_definition_text(definition=definition, op_type=op_type, language=language).strip()
    fmt = get_code_format_text(language=language, target_gpu=target_gpu).strip()
    return (
        f"{base}\n\nTarget GPU: {target_gpu}\n\n"
        f"{fmt}\n"
        "Return only the XML blocks."
    )


def get_optimization_prompt(
    *,
    definition: dict[str, Any],
    op_type: str,
    language: str,
    target_gpu: str,
    trace_logs: str,
    current_code: str,
    current_best: str | None = None,
    previous_round_summary: str | None = None,
) -> str:
    base = get_definition_text(definition=definition, op_type=op_type, language=language).strip()
    fmt = get_code_format_text(language=language, target_gpu=target_gpu).strip()
    parts: list[str] = [f"{base}\n\nTarget GPU: {target_gpu}\n"]
    parts.append(f"{fmt}")
    parts.append("\nCurrent implementation:\n" + str(current_code or "").strip())
    if previous_round_summary:
        parts.append("\nPrevious round summary:\n" + str(previous_round_summary).strip())
    if trace_logs:
        parts.append("\nExecution log / feedback:\n" + str(trace_logs).strip())
    if current_best:
        parts.append("\nCurrent best:\n" + str(current_best).strip())
    parts.append(
        "\nBefore changing the code: briefly analyze bottlenecks and correctness risks (from logs), "
        "fix compilation/correctness first, then output improved XML blocks."
    )
    return "\n\n".join([p for p in parts if p.strip()])
