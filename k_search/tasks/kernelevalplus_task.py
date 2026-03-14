"""KernelEvalPlus task adapter.

Integrates kernelevalplus-main's unified_test_runner into k-search Task protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)
from k_search.tasks.kernelevalplus_prompts import (
    get_code_format_text as ke_get_code_format_text,
    get_definition_text as ke_get_definition_text,
    get_generation_prompt as ke_get_generation_prompt,
    get_optimization_prompt as ke_get_optimization_prompt,
)
from k_search.utils.paths import get_ksearch_artifacts_dir


@dataclass(frozen=True)
class KernelEvalPlusTaskConfig:
    definition_path: Path
    kernelevalplus_root: Path
    artifacts_dir: Optional[str] = None
    name: str = "kernelevalplus"


class KernelEvalPlusTask:
    """Task wrapper around KernelEvalPlus unified_test_runner."""

    def __init__(
        self,
        *,
        definition_path: str | Path,
        kernelevalplus_root: str | Path,
        artifacts_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        def_path = Path(definition_path).expanduser().resolve()
        root = Path(kernelevalplus_root).expanduser().resolve()
        if not def_path.exists():
            raise FileNotFoundError(f"Definition JSON not found: {def_path}")
        if not root.exists():
            raise FileNotFoundError(f"KernelEvalPlus root not found: {root}")

        self._cfg = KernelEvalPlusTaskConfig(
            definition_path=def_path,
            kernelevalplus_root=root,
            artifacts_dir=(str(artifacts_dir) if artifacts_dir is not None else None),
            name=str(name or "kernelevalplus"),
        )
        self._name = self._cfg.name
        self._ksearch_artifacts_dir: str | None = self._cfg.artifacts_dir
        self._definition: dict[str, Any] = json.loads(def_path.read_text(encoding="utf-8"))
        self._op_type = str(self._definition.get("op_type", "") or "").strip()
        self._definition_name = str(self._definition.get("name", "") or def_path.stem)

        # Last-round cache for prompt feedback (best-effort).
        self._last_round_trace_logs_for_prompt: str = ""
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 0
        self._last_round_summary_line: str = ""

        # Lazy-init runner (imports are heavy).
        self._runner = None

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_cli_args(
        cls,
        *,
        definition_path: str,
        kernelevalplus_root: str,
        artifacts_dir: str | None,
    ) -> "KernelEvalPlusTask":
        return cls(
            definition_path=definition_path,
            kernelevalplus_root=kernelevalplus_root,
            artifacts_dir=artifacts_dir,
            name="kernelevalplus",
        )

    def _ensure_runner(self):
        if self._runner is not None:
            return self._runner
        # Inject KernelEvalPlus root into sys.path to import unified_test_runner.
        if str(self._cfg.kernelevalplus_root) not in sys.path:
            sys.path.insert(0, str(self._cfg.kernelevalplus_root))
        from llm_kernel_test.unified_test_runner import UnifiedTestRunner  # type: ignore

        self._runner = UnifiedTestRunner()
        return self._runner

    def _safe_name(self, s: str) -> str:
        return "".join([c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(s or "")])

    def _extract_kernel_cu_from_raw(self, raw: Any) -> str:
        if isinstance(raw, dict):
            return str(raw.get("kernel.cu", "") or "")
        s = str(raw or "")
        m = re.search(r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>', s, re.DOTALL)
        if m:
            return (m.group(1) or "").strip()
        return s

    def get_definition_text(self, language: str | None = None) -> str:
        return ke_get_definition_text(
            definition=self._definition,
            op_type=self._op_type,
            language=language,
        )

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        return ke_get_code_format_text(language=language, target_gpu=target_gpu)

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        return ke_get_generation_prompt(
            definition=self._definition,
            op_type=self._op_type,
            language=language,
            target_gpu=target_gpu,
        )

    def get_optimization_prompt(
        self,
        *,
        language: str,
        target_gpu: str,
        trace_logs: str,
        current_code: str,
        current_best: str | None = None,
        previous_round_summary: str | None = None,
    ) -> str:
        return ke_get_optimization_prompt(
            definition=self._definition,
            op_type=self._op_type,
            language=language,
            target_gpu=target_gpu,
            trace_logs=trace_logs,
            current_code=current_code,
            current_best=current_best,
            previous_round_summary=previous_round_summary,
        )

    def get_solution(self, solution_name: str) -> Solution | None:
        name = str(solution_name)
        # Allow resolving from k-search artifacts Solution JSON (by path or by name).
        try:
            d = load_ksearch_solution_json(
                solution_ref=name,
                definition_name=str(self.name or ""),
                artifacts_dir=self._ksearch_artifacts_dir,
            )
            sol = solution_from_json_dict(d)
            if str(sol.definition or "") != str(self.name or ""):
                return None
            return sol
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        lang = str(language or "").strip().lower()
        if lang != "cuda":
            raise ValueError("KernelEvalPlusTask only supports CUDA language.")
        sol_name = f"{model_name}_{self._definition_name}_r{int(round_num)}"

        # cleaned_code for CUDA is expected to be a dict with kernel.cu, kernel.h, main.cpp.
        if isinstance(cleaned_code, dict):
            files = {str(k): str(v) for k, v in cleaned_code.items()}
        else:
            files = {"kernel.cu": self._extract_kernel_cu_from_raw(raw_code)}

        sources = [
            SourceFile(path="kernel.h", content=str(files.get("kernel.h", "") or "")),
            SourceFile(path="kernel.cu", content=str(files.get("kernel.cu", "") or "")),
            SourceFile(path="main.cpp", content=str(files.get("main.cpp", "") or "")),
        ]
        spec = BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[str(target_gpu)],
            entry_point="main.cpp::run",
        )
        return Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=spec,
            sources=sources,
            description=f"KernelEvalPlus CUDA kernel for {self._definition_name}",
        )

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        lang = str(language or "").strip().lower()
        if lang != "cuda":
            return str(raw or "")
        return self._extract_kernel_cu_from_raw(raw)

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=None)

    def _attempt_dir_for_solution(self, solution: Solution, round_num: int | None) -> Path:
        root = get_ksearch_artifacts_dir(base_dir=self._ksearch_artifacts_dir, task_name=self._name)
        safe_sol = self._safe_name(str(solution.name or "solution"))
        rn = f"r{int(round_num)}" if round_num is not None else "r?"
        return root / "kernelevalplus" / "attempts" / f"{safe_sol}_{rn}"

    def _build_log_excerpt(self, results: Dict[str, Any]) -> str:
        parts: list[str] = []
        comp = results.get("compilation", {}) if isinstance(results, dict) else {}
        corr = results.get("correctness", {}) if isinstance(results, dict) else {}
        perf = results.get("performance", {}) if isinstance(results, dict) else {}
        if isinstance(comp, dict) and comp.get("errors"):
            parts.append("Compilation errors:\n" + "\n".join(comp.get("errors", [])[:5]))
        if isinstance(corr, dict) and corr.get("errors"):
            parts.append("Correctness errors:\n" + "\n".join(corr.get("errors", [])[:5]))
        if isinstance(perf, dict) and perf.get("errors"):
            parts.append("Performance errors:\n" + "\n".join(perf.get("errors", [])[:5]))
        return "\n\n".join([p for p in parts if p.strip()])

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        entry_src = None
        for sf in solution.sources or []:
            if sf.path == "kernel.cu":
                entry_src = sf
                break
        if entry_src is None:
            entry_src = solution.get_entry_source()

        kernel_cu = (entry_src.content if entry_src else "") or ""
        if not str(kernel_cu).strip():
            return EvalResult(
                status="failed",
                latency_ms=None,
                reference_latency_ms=None,
                mean_vs_baseline_factor=None,
                speedup_factor=None,
                log_excerpt="kernel.cu is empty or missing",
                metrics={"score_name": "performance_ratio", "score": None},
            )

        attempt_dir = self._attempt_dir_for_solution(solution, round_num)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "kernel.cu").write_text(str(kernel_cu), encoding="utf-8")

        runner = self._ensure_runner()
        results = runner.test(
            attempt_id=str(solution.name or "attempt"),
            variant="default",
            attempt_path=str(attempt_dir),
            definition_path=str(self._cfg.definition_path),
        )

        comp = results.get("compilation", {}) if isinstance(results, dict) else {}
        corr = results.get("correctness", {}) if isinstance(results, dict) else {}
        perf = results.get("performance", {}) if isinstance(results, dict) else {}

        comp_ok = bool(comp.get("success", False)) if isinstance(comp, dict) else False
        corr_ok = bool(corr.get("passed", False)) if isinstance(corr, dict) else False

        # Determine performance metric and score.
        latency_ms = None
        perf_value = None
        perf_metric = None
        perf_ratio = None
        benchmarks = perf.get("benchmarks") if isinstance(perf, dict) else None
        if isinstance(benchmarks, list) and benchmarks:
            b0 = benchmarks[0]
            if isinstance(b0, dict):
                latency_ms = b0.get("latency_ms")
                for k, v in b0.items():
                    if k in ("config", "latency_ms"):
                        continue
                    if isinstance(v, (int, float)):
                        perf_metric = str(k)
                        perf_value = float(v)
                        break
        bc = perf.get("baseline_comparison") if isinstance(perf, dict) else None
        if isinstance(bc, dict):
            pr = bc.get("performance_ratio")
            if isinstance(pr, (int, float)):
                perf_ratio = float(pr)

        score_name = None
        score_val = None
        if perf_ratio is not None:
            score_name = "performance_ratio"
            score_val = perf_ratio / 100.0
        elif perf_value is not None:
            score_name = perf_metric or "performance"
            score_val = perf_value

        status = "passed" if (comp_ok and corr_ok and score_val is not None) else "failed"

        # Update last-round caches for prompt feedback.
        try:
            test_cases = corr.get("test_cases") if isinstance(corr, dict) else None
            if isinstance(test_cases, list) and test_cases:
                total = len(test_cases)
                passed = len([t for t in test_cases if isinstance(t, dict) and t.get("passed")])
            else:
                total = 1
                passed = 1 if corr_ok else 0
            self._last_round_total_workloads = int(total)
            self._last_round_passed_count = int(passed)
            score_text = f"{score_name}={score_val:.4f}" if isinstance(score_val, (int, float)) else "-"
            self._last_round_summary_line = (
                f"[kernelevalplus] Round {round_num if round_num is not None else '?'}: "
                f"tests={passed}/{total} | status={status} | latency_ms={latency_ms} | {score_text}"
            )
            self._last_round_trace_logs_for_prompt = self._build_log_excerpt(results)
        except Exception:
            pass

        return EvalResult(
            status=status,
            latency_ms=(float(latency_ms) if isinstance(latency_ms, (int, float)) else None),
            reference_latency_ms=None,
            mean_vs_baseline_factor=(perf_ratio / 100.0 if isinstance(perf_ratio, (int, float)) else None),
            speedup_factor=None,
            log_excerpt=self._build_log_excerpt(results),
            metrics={
                "score_name": str(score_name or "performance"),
                "score": score_val,
                "performance_metric": perf_metric,
                "performance_value": perf_value,
                "performance_ratio": perf_ratio,
                "definition": self._definition_name,
                "op_type": self._op_type,
            },
        )

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for sol in solutions or []:
            if sol is None:
                continue
            er = self.run_benchmark(solution=sol, dump_traces=False, round_num=None)
            out.append(
                {
                    "solution": str(getattr(sol, "name", "") or ""),
                    "status": str(er.status or ""),
                    "latency_ms": er.latency_ms,
                    "score_name": (er.metrics.get("score_name") if isinstance(er.metrics, dict) else None),
                    "score": (er.metrics.get("score") if isinstance(er.metrics, dict) else None),
                }
            )
        return {
            "task": str(self._name),
            "definition": self._definition_name,
            "op_type": self._op_type,
            "solutions": out,
        }

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return str(getattr(self, "_last_round_trace_logs_for_prompt", "") or "")

    def get_last_round_passed_count(self) -> int:
        try:
            return int(getattr(self, "_last_round_passed_count", 0) or 0)
        except Exception:
            return 0

    def get_last_round_total_workloads(self) -> int:
        try:
            return int(getattr(self, "_last_round_total_workloads", 0) or 0)
        except Exception:
            return 0

    def get_last_round_summary_line(self) -> str:
        return str(getattr(self, "_last_round_summary_line", "") or "")

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_type": "kernelevalplus",
            "definition_path": str(self._cfg.definition_path),
            "definition_name": self._definition_name,
            "op_type": self._op_type,
            "kernelevalplus_root": str(self._cfg.kernelevalplus_root),
        }
