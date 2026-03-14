#!/usr/bin/env python3
"""
KernelEvalPlus Bench - Streamlit Web UI - V2
Complete rewrite with improved baseline comparison and proper batch size support
"""
import os
import re
import subprocess
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="KernelEvalPlus Bench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .case-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    .case-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #e0e7ff;
        color: #4f46e5;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .tag-variant {
        background-color: #fef3c7;
        color: #d97706;
    }
    .tag-model {
        background-color: #d1fae5;
        color: #059669;
    }
    .tag-layer {
        background-color: #dbeafe;
        color: #2563eb;
    }
    .tag-quant {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
    }
    .tag-model-name {
        background-color: #fef3c7;
        color: #d97706;
        font-weight: 600;
    }
    .tag-layer-type {
        background-color: #e0f2fe;
        color: #0284c7;
        font-weight: 600;
    }
    .hardware-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #f3f4f6;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFINITIONS_DIR = PROJECT_ROOT / "definitions" / "quant_gemm"
RESULTS_ROOT_DIR = PROJECT_ROOT / "llm_kernel_test" / "results"
BATCH_RUNS_DIR = RESULTS_ROOT_DIR / "batch_runs"
SANDBOX_GENERATED_DIR = PROJECT_ROOT / "llm_kernel_test" / "sandbox" / "generated"
BASELINE_FILE = PROJECT_ROOT / "data" / "baseline" / "baseline_data_compact.json"

def detect_local_gpu_name() -> str:
    """Best-effort local GPU name detection (used as a display label)."""
    env_hw = os.environ.get("KEVAL_HARDWARE")
    if env_hw:
        return env_hw

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        name = out.strip().splitlines()[0].strip() if out else None
        return name or "Unknown"
    except Exception:
        return "Unknown"

CURRENT_HARDWARE = detect_local_gpu_name()

# Baseline hardware priority mapping
BASELINE_PRIORITY = {
    "laptop": "RTX4070",      # RTX 4070 Laptop, RTX 5070 Laptop
    "desktop": "RTX4090",     # RTX 4090 Desktop
    "server": "H800",         # H800 Server
}

# Hardware type detection keywords
HARDWARE_TYPE_KEYWORDS = {
    "laptop": ["laptop", "mobile", "notebook", "4070", "5070"],  # RTX 4070/5070 are laptop GPUs
    "desktop": ["4090", "rtx", "gtx", "geforce"],
    "server": ["a100", "h100", "h800", "a40", "a30", "a10", "l40", "v100"],
}

def detect_hardware_type(hardware_name: str) -> str:
    """Detect hardware type from name (laptop, desktop, server)."""
    hardware_lower = hardware_name.lower()
    for hw_type, keywords in HARDWARE_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in hardware_lower:
                return hw_type
    return "desktop"  # Default fallback

def get_baseline_hardware_for_comparison(local_hardware: str) -> str:
    return BASELINE_PRIORITY.get(detect_hardware_type(local_hardware), "RTX4090")

def extract_nk_from_filename(name):
    """Extract N, K dimensions from filename."""
    match = re.search(r'n(\d+)_k(\d+)', name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def extract_quant_type(name):
    """Extract quantization type from filename (e.g., w4a32c8, w8a32c8)."""
    match = re.match(r'([wh]\d+[aA]\d+[cC]?\d*)', name)
    if match:
        return match.group(1).lower()
    return None

def extract_weight_quant(name: str):
    """Extract weight quantization (q4_0/q4_1/q8_0) from any identifier string."""
    s = (name or "").lower()
    for q in ("q4_0", "q4_1", "q8_0"):
        if q in s:
            return q
    return None

def compilation_success(test_result: dict) -> bool:
    if isinstance(test_result, dict) and 'compilation' in test_result:
        comp = test_result.get('compilation', {})
        if isinstance(comp, dict):
            return bool(comp.get('success', False))
        if isinstance(comp, bool):
            return comp
    return False

def correctness_passed(test_result: dict) -> bool:
    corr = test_result.get('correctness', {})
    if isinstance(corr, dict):
        return bool(corr.get('passed', False))
    if isinstance(corr, bool):
        return corr
    return False

def get_all_quant_types(test_cases):
    """Get all unique quantization types from test cases."""
    quant_types = set()
    for case_name, case_data in test_cases.items():
        qtype = extract_quant_type(case_name)
        if not qtype:
            hw_results = case_data.get('results_by_hardware', {}).get(CURRENT_HARDWARE, [])
            if hw_results:
                qtype = extract_quant_type(hw_results[0].get('variant', ''))
        if qtype:
            quant_types.add(qtype)
    return sorted(quant_types)

def extract_model_from_case(case_name, case_data):
    """Extract model name from case."""
    info = case_data.get('info', {})
    model_archs = info.get('model_architectures', [])
    if model_archs:
        return model_archs[0]

    name_lower = case_name.lower()
    model_patterns = {
        'deepseek-v3': ['ds3', 'deepseek_v3', 'deepseek-v3'],
        'deepseek-v2': ['ds2', 'deepseek_v2', 'deepseek-v2'],
        'llama-3.1-8b': ['llama3_8b', 'llama-3-8b', 'llama-3-8b'],
        'llama-3.1-70b': ['llama3_70b', 'llama3_70b', 'llama-3-70b'],
        'llama-3': ['llama3'],
        'qwen3': ['qwen3'],
        'qwen2.5-7b': ['qwen2_5_7b', 'qwen2.5', 'qwen2.5-7b'],
        'mistral': ['mistral'],
        'mixtral': ['mixtral'],
    }

    for model_name, patterns in model_patterns.items():
        for pattern in patterns:
            if pattern in name_lower:
                return model_name
    return "Other"

def get_all_models(test_cases):
    """Get all unique models from test cases."""
    models = {}
    for case_name, case_data in test_cases.items():
        model = extract_model_from_case(case_name, case_data)
        if model not in models:
            models[model] = 0
        models[model] += 1
    # Sort by count desc, then by name
    return sorted(models.keys(), key=lambda x: (-models[x], x))

def extract_layer_from_case(case_name, case_data):
    """Extract layer type from case."""
    info = case_data.get('info', {})
    tags = info.get('tags', [])

    for tag in tags:
        if "layer:" in tag:
            return tag.replace("layer:", "").strip()

    name_lower = case_name.lower()
    layer_patterns = {
        'att_out': ['att_out', 'attention-output', 'att.out'],
        'att_qkv': ['att_qkv', 'attention-qkv', 'att.qkv', 'qkv'],
        'lm_head': ['lm_head', 'language.model', 'lm.head'],
        'ffn_up': ['ffn_up', 'feed-forward', 'ffn.up'],
        'ffn_down': ['ffn_down', 'feed-forward.down'],
        'moe_up': ['moe_up', 'moe.up'],
        'moe_down': ['moe_down', 'moe.down'],
        'att': ['att_'],
    }

    for layer_name, patterns in layer_patterns.items():
        for pattern in patterns:
            if pattern in name_lower:
                return layer_name
    return "Other"

def get_all_layers(test_cases):
    """Get all unique layer types from test cases."""
    layers = {}
    for case_name, case_data in test_cases.items():
        layer = extract_layer_from_case(case_name, case_data)
        if layer not in layers:
            layers[layer] = 0
        layers[layer] += 1
    return sorted(layers.keys(), key=lambda x: (-layers[x], x))

def get_definition_info(definition_path):
    """Get definition info from JSON file."""
    try:
        p = Path(definition_path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
    except:
        pass

    return {}

# ============================================
# BASELINE COMPARISON - NEW IMPLEMENTATION
# ============================================

def load_baseline_data():
    """Load GGML baseline data."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}

def find_baseline_for_benchmark(bench, weight_quant):
    """Find baseline data for a specific benchmark.

    Baseline key format: w{w|8}a32c8_{q4_0|q8_0}_f32_m{K_in}_n{batch}_k{N_out}
    Where:
    - K_in = input features (what we multiply with weight)
    - N_out = output features (result size)
    - batch = batch size (M in GGML terms)

    Test result format:
    - bench['M']: batch size (e.g., single_token means M=1)
    - bench['N']: output features (e.g., 7168)
    - bench['K']: input features (e.g., 7168)

    Mapping to baseline:
    - bench['M'] → baseline batch size (n)
    - bench['N'] → baseline input features (m)
    - bench['K'] → baseline output features (k)
    """
    if not bench:
        return None

    m_b = bench.get('M', 0)
    n_b = bench.get('N', 0)
    k_b = bench.get('K', 0)

    # Get quant type for baseline lookup
    # Baseline uses w4a32c8 for Q4_0 and w8a32c8 for Q8_0
    baseline_type = "w4a32c8_q4_0_f32"  # Default to Q4_0

    # Try the correct baseline key format
    # Baseline: m{K_in}_n{batch}_k{N_out}
    # For single_token (M=1, N=7168, K=7168):
    #   baseline key: w4a32c8_q4_0_f32_m7168_n1_k7168
    #   This maps to: m=7168, n=1, k=7168

    baseline_prefix = "w8a32c8" if weight_quant == "q8_0" else "w4a32c8"
    case_id = f"{baseline_prefix}_f32_m{k_b}_n{m_b}_k{n_b}"

    baseline_data = load_baseline_data()
    baseline_entry = baseline_data.get(case_id)

    if baseline_entry:
        hw = baseline_entry.get('hardware', {})
        baseline_hw = get_baseline_hardware_for_comparison(CURRENT_HARDWARE)
        baseline_hw_data = hw.get(baseline_hw, {})

        if baseline_hw_data:
            gflops = baseline_hw_data.get('gflops', 0)
            baseline_gflops = baseline_hw_data.get('tflops', 0)

            return {
                'gflops': gflops,
                'tflops': baseline_gflops,
                'hw': baseline_hw,
            }

    return None

# ============================================
# DATA LOADING
# ============================================

def load_batch_results():
    """Load batch results from multiple on-disk layouts."""
    batches = []

    # 1) New: batch_runs/<run>/*_results.json (+ metadata.json)
    if BATCH_RUNS_DIR.exists():
        for run_dir in BATCH_RUNS_DIR.iterdir():
            if not run_dir.is_dir():
                continue

            meta = {}
            meta_file = run_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}

            timestamp = meta.get("timestamp")
            if not timestamp:
                try:
                    timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()
                except Exception:
                    timestamp = ""

            run_results = []
            for result_file in sorted(run_dir.glob("*_results.json")):
                try:
                    with open(result_file) as f:
                        tr = json.load(f)
                except Exception:
                    continue

                attempt_id = tr.get("attempt_id") or result_file.name.replace("_results.json", "")
                compilation_ok = compilation_success(tr)
                correctness_ok = correctness_passed(tr)

                perf = tr.get('performance', {})
                benchmarks = perf.get('benchmarks', [])

                run_results.append({
                    "definition": attempt_id,
                    "attempt_id": attempt_id,
                    "success": compilation_ok and correctness_ok,
                    "test_result": tr,
                    "provider": meta.get("provider", ""),
                    "model": meta.get("model", ""),
                    "timestamp": timestamp,
                })

            if not run_results:
                continue

            total = len(run_results)
            success = sum(1 for r in run_results if r.get('success'))

            batches.append({
                "batch_name": run_dir.name,
                "folder": run_dir.name,
                "timestamp": timestamp,
                "provider": meta.get("provider", ""),
                "model": meta.get("model", ""),
                "total": total,
                "success": success,
                "results": run_results,
                "source": "batch_runs",
            })

    # 2) Legacy: results/*_results.json (flat files)
    legacy_files = []
    if RESULTS_ROOT_DIR.exists():
        for rf in RESULTS_ROOT_DIR.glob("*_results.json"):
            if rf.parent.name == "batch_runs":
                continue

            legacy_files.append(rf)

    if legacy_files:
        try:
            latest_ts = ""
            results = []

            for rf in sorted(legacy_files):
                try:
                    with open(rf) as f:
                        tr = json.load(f)
                except Exception:
                    continue

                attempt_id = tr.get("attempt_id") or rf.name.replace("_results.json", "")
                compilation_ok = compilation_success(tr)
                correctness_ok = correctness_passed(tr)
                perf = tr.get("performance", {})

                tested_at = tr.get("tested_at", "")
                if tested_at and tested_at > latest_ts:
                    latest_ts = tested_at

                run_results.append({
                    "definition": attempt_id,
                    "attempt_id": attempt_id,
                    "success": compilation_ok and correctness_ok,
                    "test_result": tr,
                })

            total = len(run_results)
            success = sum(1 for r in run_results if r.get('success'))

            batches.append({
                "batch_name": "results_root",
                "folder": "results_root",
                "timestamp": latest_ts,
                "provider": "",
                "model": "",
                "total": total,
                "success": success,
                "results": run_results,
                "source": "results_root",
            })
        except Exception:
            pass

    # Sort by timestamp
    batches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return batches

def organize_test_cases(batch_results):
    """Organize test results by case (definition file)."""
    cases = {}

    for batch in batch_results:
        for r in batch.get('results', []):
            definition = r.get('definition', '')
            definition_name = definition.split('/')[-1] if '/' in definition else definition

            if definition_name not in cases:
                # Load definition info
                def_info = get_definition_info(definition)

                cases[definition_name] = {
                    'name': definition_name,
                    'definition_path': definition,
                    'info': def_info,
                    'variants': {},
                    'results_by_hardware': {},
                }

            # Get variant info
            attempt_id = r.get('attempt_id', '')
            variant = r.get('test_result', {}).get('variant', '')

            # Store by hardware label (single-device by default; multi-device can be added later)
            hw_key = CURRENT_HARDWARE
            if hw_key not in cases[definition_name]['results_by_hardware']:
                cases[definition_name]['results_by_hardware'][hw_key] = []

            # Extract performance data
            perf = r.get('test_result', {}).get('performance', {})
            benchmarks = perf.get('benchmarks', [])

            # Compile result metrics
            nmse = None
            corr_obj = r.get('test_result', {}).get('correctness', {})
            if isinstance(corr_obj, dict):
                nmse = corr_obj.get("nmse", None)

            result_data = {
                'batch': batch['batch_name'],
                'timestamp': batch.get('timestamp', ''),
                'provider': batch.get('provider', ''),
                'model': batch.get('model', ''),
                'attempt_id': attempt_id,
                'variant': variant,
                'success': r.get('success', False),
                'compilation': compilation_success(r),
                'correctness': correctness_passed(r),
                'nmse': nmse,
                'benchmarks': benchmarks,
            }

            cases[definition_name]['results_by_hardware'][hw_key].append(result_data)
            cases[definition_name]['variants'][variant] = attempt_id

    return cases

def render_case_card(case_name, case_data, baseline_data, show_baseline=True):
    """Render a test case card with baseline comparison."""
    info = case_data.get('info', {})
    tags = info.get('tags', [])

    # Get latest result for current hardware
    hw_results = case_data.get('results_by_hardware', {}).get(CURRENT_HARDWARE, [])
    if not hw_results:
        return None

    latest = hw_results[0]

    # Get benchmarks
    perf = latest.get('benchmarks', [])

    # Extract quant type and weight quant
    quant_type = extract_quant_type(case_name) or extract_quant_type(latest.get('variant', ''))
    weight_quant = extract_weight_quant(case_name) or extract_weight_quant(latest.get('variant', '')) or 'q4_0'

    # Get N, K from benchmark if available in file
    n, k = None, None
    for bench in perf:
        if 'N' in bench and 'K' in bench:
            n = bench['N']
            k = bench['K']
            break

    # Fallback: extract from filename
    if (not n or not k) and perf:
        n, k = extract_nk_from_filename(case_name)

    # Find baseline for each benchmark
    baseline_info = None
    baseline_ratios = []

    for bench in perf:
        baseline = find_baseline_for_benchmark(bench, weight_quant)

        if baseline and baseline.get('gflops'):
            current_gflops = bench.get('gflops', 0)
            ratio = (current_gflops / baseline['gflops']) * 100 if baseline['gflops'] > 0 else 0

            baseline_ratios.append({
                'shape': bench.get('shape', ''),
                'M': bench.get('M', 0),
                'gflops': current_gflops,
                'baseline_gflops': baseline['gflops'],
                'ratio': ratio,
                'baseline_hw': baseline['hw'],
            })

    # Build tags HTML
    model = extract_model_from_case(case_name, case_data)
    layer = extract_layer_from_case(case_name, case_data)

    tags_html = f'<span class="tag tag-model-name">🤖 {model}</span>'
    tags_html += f'<span class="tag tag-layer-type">📚 {layer}</span>'

    if quant_type:
        tags_html += f'<span class="tag tag-quant">⚡ {quant_type}</span>'

    for tag in tags:
        if "variant:" in tag:
            tags_html += f'<span class="tag tag-variant">{tag}</span>'

    # Metrics
    metrics = []
    for bl in baseline_ratios:
        metrics.append({
            'label': f"vs {bl['baseline_hw']}",
            'gflops': bl['gflops'],
            'baseline_gflops': bl['baseline_gflops'],
            'ratio': bl['ratio'],
            'shape': bl['shape'],
        })

    # Display
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

    name_display = case_name.replace('.json', '')[:60]

    with col1:
        st.markdown(f"**{name_display}**")
        if tags_html:
            st.markdown(tags_html, unsafe_allow_html=True)

    with col2:
        if n and k:
            st.caption(f"N={int(n):,}, K={int(k):,}")

    with col3:
        if metrics:
            for m in metrics[:3]:  # Show top 3
                st.metric(m['label'], f"{m['gflops']:.1f} / {m['baseline_gflops']:.1f}",
                       help=f"{m['shape']} vs {m['baseline_hw']}")
        if len(metrics) > 3:
            st.caption(f"+{len(metrics)-3} more")

    with col4:
        if latest.get('success'):
            st.success("✅ Pass")
        else:
            st.error("❌ Fail")

    if show_baseline:
        with st.expander("Baseline Comparison", expanded=False):
            st.markdown("| Shape | GFLOPS | Baseline | Ratio | HW")
            for bl in baseline_ratios:
                hw_label = bl['baseline_hw'].upper()
                st.markdown(f"| {bl['shape']:8} | **{bl['gflops']:.1f}** / **{bl['baseline_gflops']:.1f}** | {bl['ratio']:.1f}% | {hw_label}")

def main():
    # Sidebar
    st.sidebar.markdown("## ⚡ KernelEvalPlus Bench")
    st.sidebar.markdown("---")

    # Load data
    batch_results = load_batch_results()
    baseline_data = load_baseline_data()
    test_cases = organize_test_cases(batch_results)

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "📋 Test Cases", "🏆 Performance Leaderboard", "🧪 Models", "📄 Definition Viewer"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Hardware")
    st.sidebar.info(f"**Current:** {CURRENT_HARDWARE}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **KernelEvalPlus** - LLM-driven CUDA kernel generation and testing.

    Inspired by [FlashInfer Bench](https://bench.flashinfer.ai)
    """)

    # Page: Dashboard
    if page == "🏠 Dashboard":
        st.markdown('<h1 class="main-header">Dashboard</h1>', unsafe_allow_html=True)

        if not batch_results:
            st.info("No batch test results found.")
            return

        # Latest batch summary
        latest = batch_results[0]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tests", latest.get('total', 0))
        with col2:
            st.metric("Passed", latest.get('success', 0), delta_color="normal")
        with col3:
            pass_rate = (latest.get('success', 0) / latest.get('total', 1) * 100)
            st.metric("Pass Rate", f"{pass_rate:.0f}%")
        with col4:
            st.metric("Test Cases", len(test_cases))

        st.markdown("---")

        # Hardware info cards
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"🖥️ Test Hardware: {CURRENT_HARDWARE}")
        st.caption("Set `KEVAL_HARDWARE` to override")

        with col2:
            baseline_hw = []
            if baseline_data:
                first_key = list(baseline_data.keys())[0]
                for hw in baseline_data.get(first_key, {}).get('hardware', {}):
                    if hw:
                        baseline_hw.append(hw)

            if baseline_hw:
                st.info(" | ".join(sorted(set(baseline_hw))))
            else:
                st.warning("No baseline data found")

        st.markdown("---")

        # Latest batch results
        st.subheader("📋 Recent Batch Runs")
        results_data = []
        for batch in batch_results[:5]:
            ts = batch.get('timestamp', '')
            try:
                ts_display = datetime.fromisoformat(ts).strftime("%m-%d %H:%M") if ts else "-"
            except Exception:
                ts_display = ts[:16] if isinstance(ts, str) and ts else "-"

            results_data.append({
                "Batch": batch['batch_name'][:30],
                "Time": ts_display,
                "Provider": batch.get('provider', '-'),
                "Model": batch.get('model', '-').replace('deepseek-v3.2', 'DS-v3.2'),
                "Total": batch.get('total', 0),
                "Success": batch.get('success', 0),
                "Pass Rate": f"{batch.get('success', 0) / batch.get('total', 1) * 100:.0f}%" if batch.get('total', 0) > 0 else "0"
            })

        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Page: Test Cases
    elif page == "📋 Test Cases":
        st.markdown('<h1 class="main-header">Test Cases</h1>', unsafe_allow_html=True)

        st.info("📌 Click on a case to see detailed performance comparison across hardware and baselines.")
        st.markdown("---")

        # Get all filter options
        all_quant_types = get_all_quant_types(test_cases)
        all_models = get_all_models(test_cases)
        all_layers = get_all_layers(test_cases)

        # Filters - Row 1: Model, Layer, Quant Type
        col1, col2, col3 = st.columns(3)
        with col1:
            model_filter = st.multiselect("🤖 Model", all_models, default=all_models, placeholder="Select models...")
        with col2:
            layer_filter = st.multiselect("📚 Layer", all_layers, default=all_layers, placeholder="Select layers...")
        with col3:
            quant_filter = st.multiselect("⚡ Quant Type", all_quant_types, default=all_quant_types, placeholder="Select quant types...")

        # Filters - Row 2: Search, Sort, Status
        col1, col2, col3 = st.columns(3)
        with col1:
            search = st.text_input("🔍 Search", placeholder="Search by name, model, layer...")
        with col2:
            sort_by = st.selectbox("Sort", ["Name", "GFLOPS", "Dimensions"])
        with col3:
            status_filter = st.selectbox("Status", ["All", "Passed Only", "Failed"])

        # Filter and sort cases
        filtered_cases = []
        for case_name, case_data in test_cases.items():
            # Search filter
            if search and search.lower() not in case_name.lower():
                continue

            # Quant type filter
            case_quant = extract_quant_type(case_name)
            if quant_filter and case_quant not in quant_filter:
                continue

            # Model filter
            case_model = extract_model_from_case(case_name, case_data)
            if model_filter and case_model not in model_filter:
                continue

            # Layer filter
            case_layer = extract_layer_from_case(case_name, case_data)
            if layer_filter and case_layer not in layer_filter:
                continue

            # Status filter
            hw_results = case_data.get('results_by_hardware', {}).get(CURRENT_HARDWARE, [])
            if not hw_results:
                continue

            latest = hw_results[0]

            if status_filter == "Passed Only" and not latest.get('success'):
                continue
            elif status_filter == "Failed" and latest.get('success'):
                continue

            filtered_cases.append((case_name, case_data))

        # Sort
        if sort_by == "Name":
            filtered_cases.sort(key=lambda x: x[0])
        elif sort_by == "GFLOPS":
            filtered_cases.sort(key=lambda x: x[1].get('results_by_hardware', {}).get(CURRENT_HARDWARE, [{}])[0].get('benchmarks', []).__iter__().next(lambda x: x[1].get('gflops', 0) if x else 0, reverse=True), reverse=False)
        elif sort_by == "Dimensions":
            filtered_cases.sort(key=lambda x: extract_nk_from_filename(x[0])[0] or 0, reverse=True)
        else:
            filtered_cases.sort(key=lambda x: x[0])

        # Case count
        st.caption(f"Showing {len(filtered_cases)} of {len(test_cases)} test cases")

        # Render cases
        for case_name, case_data in filtered_cases:
            with st.container():
                render_case_card(case_name, case_data, baseline_data)
                st.markdown("---")

    # Page: Performance Leaderboard
    elif page == "🏆 Performance Leaderboard":
        st.markdown('<h1 class="main-header">Performance Leaderboard</h1>', unsafe_allow_html=True)

        st.info("💡 Performance data is now organized by batch size (M value). Click on a case card to see baseline comparison for each benchmark shape.")

        if not test_cases:
            st.info("No test cases found.")
            return

        # Collect all performance data
        all_perf_data = []

        for case_name, case_data in test_cases.items():
            hw_results = case_data.get('results_by_hardware', {}).get(CURRENT_HARDWARE, [])
            if hw_results:
                latest = hw_results[0]
                if latest.get('success'):
                    perf = latest.get('benchmarks', [])

                    for bench in perf:
                        m_b = bench.get('M', 0)
                        gflops = bench.get('gflops', 0)

                        # Get baseline for this benchmark
                        baseline = find_baseline_for_benchmark(bench, extract_weight_quant(case_name))

                        all_perf_data.append({
                            'Case': case_name[:50].replace('.json', ''),
                            'Shape': bench.get('shape', ''),
                            'M': m_b,
                            'GFLOPS': gflops,
                            'Baseline GFLOPS': baseline['gflops'] if baseline else None,
                            'Baseline HW': baseline['hw'] if baseline else None,
                            'Baseline Ratio %': baseline['ratio'] if baseline else None,
                            'Hardware': CURRENT_HARDWARE,
                        })

        if not all_perf_data:
            st.info("No performance data available.")
            return

        # Create DataFrame
        df = pd.DataFrame(all_perf_data)

        # Filters
        col1, col2 = st.columns([2, 2])
        with col1:
            m_values = sorted(df[df['M'] > 0]['M'].unique().tolist())
            shape_filter = st.multiselect(
                "Batch Size (M)",
                options=m_values,
                default=m_values[:6] if len(m_values) > 6 else m_values
            )
        with col2:
            hw_values = sorted(df['Hardware'].unique().tolist())
            hw_filter = st.multiselect(
                "Hardware",
                options=hw_values,
                default=[CURRENT_HARDWARE] if CURRENT_HARDWARE in hw_values else hw_values[:1]
            )

        # Filter by selection
        if shape_filter:
            df_filtered = df[df['M'].isin(shape_filter)]
        else:
            df_filtered = df

        if hw_filter:
            df_filtered = df_filtered[df_filtered['Hardware'].isin(hw_filter)]
        else:
            df_filtered = df_filtered

        # Display
        st.subheader(f"Performance: {shape_filter if shape_filter else 'All'} | {hw_filter if hw_filter else 'All'}")

        # Sort by GFLOPS descending
        df_sorted = df_filtered.sort_values('GFLOPS', ascending=False)

        # Display table with dynamic columns
        cols_to_show = ['Case', 'M', 'GFLOPS']

        # Check if baseline columns exist and have data
        has_baseline_gflops = 'Baseline GFLOPS' in df_sorted.columns and df_sorted['Baseline GFLOPS'].notna().any()
        has_baseline_ratio = 'Baseline Ratio %' in df_sorted.columns and df_sorted['Baseline Ratio %'].notna().any()
        has_baseline_hw = 'Baseline HW' in df_sorted.columns and df_sorted['Baseline HW'].notna().any()

        if has_baseline_gflops:
            cols_to_show = ['Case', 'M', 'GFLOPS', 'Baseline GFLOPS']
            if has_baseline_ratio:
                cols_to_show.append('Baseline Ratio %')
            if has_baseline_hw:
                cols_to_show.append('Baseline HW')

        st.dataframe(
            df_sorted.head(100)[cols_to_show],
            use_container_width=True,
            hide_index=True
        )

        # Chart
        if shape_filter:
            chart_data = df_sorted[df_sorted['M'].isin(shape_filter)]
        else:
            chart_data = df_sorted

        fig = px.bar(
            chart_data.head(30),
            x='GFLOPS',
            y='Case',
            orientation='h',
            title=f'Top 30 - {", ".join(map(str, shape_filter)) if shape_filter else "All"}',
            color='GFLOPS',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Page: Models
    elif page == "🧪 Models":
        st.markdown('<h1 class="main-header">Models & Architectures</h1>', unsafe_allow_html=True)

        # Group cases by model
        models = {}
        for case_name, case_data in test_cases.items():
            model = extract_model_from_case(case_name, case_data)
            if model not in models:
                models[model] = {'cases': [], 'count': 0}
            models[model]['cases'].append(case_name)
            models[model]['count'] += 1

        # Sort by count desc, then by name
        sorted_models = sorted(models.items(), key=lambda x: (-x[1]['count'], x[0]))

        for model_name, model_data in sorted_models:
            with st.expander(f"### {model_name} ({model_data['count']} test cases)", expanded=False):
                col1, col2 = st.columns([3, 1])

                # Show performance for each case
                for i, case in enumerate(model_data['cases'][:10]):
                    hw_results = case.get('results_by_hardware', {}).get(CURRENT_HARDWARE, [])
                    if hw_results and hw_results[0].get('success'):
                        perf = hw_results[0].get('benchmarks', [])
                        for bench in perf:
                            if bench.get('gflops', 0):
                                gflops = bench.get('gflops', 0)
                                st.markdown(f"- ✅ **{case[:50]}** - `{gflops:.1f}` GFLOPS")
                                break
                    # Show "... and X more" after the 10th case
                    if i == 9 and model_data['count'] > 10:
                        st.caption(f"... and {model_data['count'] - 10} more")

    # Page: Definition Viewer
    elif page == "📄 Definition Viewer":
        st.markdown('<h1 class="main-header">Kernel Definition & Solutions</h1>', unsafe_allow_html=True)
        st.caption("View kernel definitions and all LLM-generated solutions with performance comparison")

        # Scan definitions
        definitions = {}

        for category_dir in DEFINITIONS_DIR.iterdir():
            if not category_dir.is_dir() or category_dir.name == "templates":
                continue

            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                            data = json.load(f)
                            def_name = data.get('name', json_file.stem)
                            definitions[def_name] = {
                                'data': data,
                                'category': category_dir.name,
                                'file_path': str(json_file)
                            }
                except Exception:
                    pass

        # Get test results for each definition
        for def_name, def_info in definitions.items():
            def_info['solutions'] = []

            case_name = def_name
            case_data = test_cases.get(case_name, {'results_by_hardware': {}, 'info': {}})

        # Sidebar: Definition list
        st.sidebar.markdown("### 📋 Kernel Definitions")
        search_def = st.sidebar.text_input("Search...", placeholder="Search kernels...")
        model_filter = st.sidebar.multiselect("Filter by Model", get_all_models(test_cases), default=get_all_models(test_cases))
        sort_def = st.sidebar.selectbox("Sort By", ["Name", "Performance (Best GFLOPS)"])

        # Filter definitions
        filtered_defs = {}
        for def_name, def_info in definitions.items():
            if search_def and search_def.lower() not in def_name.lower():
                continue

            # Get test results for this definition
            hw_results = case_data.get('results_by_hardware', {}).get(CURRENT_HARDWARE, [])
            if not hw_results:
                continue

            latest = hw_results[0]

            # Get best performance for sorting
            best_gflops = 0
            for hw_result in hw_results:
                if hw_result.get('success'):
                    perf = hw_result.get('test_result', {}).get('performance', {}).get('benchmarks', [])
                    for bench in perf:
                        gflops = bench.get('gflops', 0)
                        if gflops > best_gflops:
                            best_gflops = gflops

            filtered_defs[def_name] = {
                'best_gflops': best_gflops,
                'info': def_info,
            }

        # Sort
        if sort_def == "Name":
            filtered_defs = dict(sorted(filtered_defs.items()))
        elif sort_def == "Performance (Best GFLOPS)":
            filtered_defs = dict(sorted(filtered_defs.items(), key=lambda x: x[1]['best_gflops'], reverse=True))
        else:
            filtered_defs = dict(sorted(filtered_defs.items()))

        # Select definition
        if filtered_defs:
            selected_def = st.sidebar.selectbox("Select Kernel", sorted(filtered_defs.keys()))
        else:
            st.warning("No definitions match your filters.")
            return

        data = definitions[selected_def]['data']
        info = data.get('info', {})
        tags = info.get('tags', [])

        # Main content - Two columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📋 Kernel Specification")
            st.markdown(f"**{data.get('name', selected_def)}**")
            if 'description' in data:
                st.caption(data.get('description', '')[:200] + "..." if len(data.get('description', '')) > 200 else data.get('description', ''))

            st.markdown("---")
            if 'axes' in data:
                st.markdown("**Dimensions:**")
                axes_html = ""
                for name, axis in data['axes'].items():
                    value = axis.get('value', 'var')
                    axes_html += f"`{name}={value}` "
                st.markdown(axes_html)
                st.markdown("**Tags:**")
                tags_display = data.get('tags', [])[:8]
                st.markdown(", ".join(f"`{tag}`" for tag in tags_display))
                if len(data.get('tags', [])) > 8:
                    st.caption(f"... and {len(data.get('tags', [])) - 8} more")

        with col2:
            st.markdown("### 📊 Solutions Overview")
            solutions = def_info['solutions']
            if not solutions:
                st.info("No test results found for this kernel definition.")
            else:
                success_count = sum(1 for s in solutions if s.get('success', False))
                st.metric("Total Solutions", len(solutions))
                st.metric("Success Rate", f"{success_count / len(solutions) * 100:.0f}%")

            # Display solutions with baseline comparison
            hw_filter = st.multiselect("Filter Hardware", [CURRENT_HARDWARE] + ([] if not baseline_data else list(set([d.get('Baseline HW', '') for d in solutions]))), default=[CURRENT_HARDWARE])

            for i, sol in enumerate(sorted(solutions, key=lambda x: (
                float(x.get('gflops', 0) if x.get('gflops', 0) else 0),
                float(x.get('baseline_gflops', 0) if x.get('baseline_gflops', 0) else 0)
            ) if x.get('success') else 0, reverse=True)[:20]):
                if not hw_filter or sol.get('Hardware') == hw_filter:
                    continue

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Show GFLOPS
                    gflops = sol.get('gflops', 0)
                    baseline = sol.get('baseline_gflops', 0)

                    if baseline:
                        ratio = (gflops / baseline * 100) if baseline > 0 else 0
                        st.markdown(f"**{gflops:.1f}** / **{baseline:.1f}** ({ratio:.1f}%)")
                    else:
                        st.markdown(f"**{gflops:.1f}** GFLOPS")

                with col2:
                    # Show variant and status
                    variant = sol.get('variant', '')
                    status = "✅ Pass" if sol.get('success') else "❌ Fail"
                    st.markdown(f"{status} {variant[:20]}")

                if i == 19:
                    st.caption(f"... and {len(solutions) - 20} more")

if __name__ == "__main__":
    main()
