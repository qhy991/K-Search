#!/usr/bin/env python3
"""
KernelEvalPlus Benchmark WebUI - CSV Version
Displays LLM-generated kernel performance from CSV format with baseline comparison.
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess

# Page config
st.set_page_config(
    page_title="KernelEvalPlus Bench",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 1rem;
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_FILE = PROJECT_ROOT / "data" / "baseline" / "baseline_data_compact.json"
CSV_FILE = PROJECT_ROOT / "data" / "experiments" / "results.csv"

# Model columns in CSV
MODEL_COLUMNS = {
    "glm-4.7-0211": {
        "tflops": "glm-4.7-0211_tflops",
        "latency": "glm-4.7-0211_latency_ms",
        "vs_baseline": "glm-4.7-0211_vs_baseline_pct",
        "name": "GLM-4.7 (0211)",
        "color": "#FF6B6B"
    },
    "glm-5-0212": {
        "tflops": "glm-5-0212_tflops",
        "latency": "glm-5-0212_latency_ms",
        "vs_baseline": "glm-5-0212_vs_baseline_pct",
        "name": "GLM-5 (0212)",
        "color": "#4ECDC4"
    },
    "sonnet-4.5": {
        "tflops": "sonnet-4.5_tflops",
        "latency": "sonnet-4.5_latency_ms",
        "vs_baseline": "sonnet-4.5_vs_baseline_pct",
        "name": "Claude Sonnet-4.5",
        "color": "#95E1D3"
    }
}

def detect_local_gpu_name() -> str:
    """Detect local GPU name.

    Hardware types:
    - Laptop: RTX 4070, RTX 5070
    - Desktop: RTX 4090
    - Server: H800, A100, etc.
    """
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
        return name or "NVIDIA GeForce RTX 4090"
    except Exception:
        return "NVIDIA GeForce RTX 4090"

CURRENT_HARDWARE = detect_local_gpu_name()

def load_baseline_data() -> Dict:
    """Load GGML baseline data."""
    if not BASELINE_FILE.exists():
        return {}
    try:
        with open(BASELINE_FILE) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading baseline: {e}")
        return {}

def load_csv_data() -> pd.DataFrame:
    """Load CSV experimental data."""
    if not CSV_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        # Convert TFLOPS to GFLOPS for display
        for model_key, model_info in MODEL_COLUMNS.items():
            tflops_col = model_info["tflops"]
            if tflops_col in df.columns:
                df[f"{model_key}_gflops"] = df[tflops_col] * 1000
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def get_baseline_hardware_list(baseline_data: Dict) -> List[str]:
    """Get list of all hardware in baseline data."""
    if not baseline_data:
        return []
    first_entry = next(iter(baseline_data.values()))
    return list(first_entry.get('hardware', {}).keys())

def parse_percentage(pct_str: str) -> float:
    """Parse percentage string like '15.2%' to float 15.2"""
    if pd.isna(pct_str):
        return 0.0
    if isinstance(pct_str, (int, float)):
        return float(pct_str)
    return float(str(pct_str).replace('%', '').strip())

def main():
    # Sidebar
    st.sidebar.markdown("## ⚡ KernelEvalPlus Bench")
    st.sidebar.markdown("---")

    # Load data
    csv_df = load_csv_data()
    baseline_data = load_baseline_data()
    baseline_hardware = get_baseline_hardware_list(baseline_data)

    if csv_df.empty:
        st.error("No CSV data found. Please check the file path.")
        st.info(f"Expected file: {CSV_FILE}")
        return

    # Get unique values
    experiments = csv_df['experiment'].unique().tolist()
    batches = sorted(csv_df['M'].unique().tolist())

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Model Comparison", "🏆 Leaderboard", "🖥️ Baseline Data", "📈 Detailed Analysis"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Hardware")
    st.sidebar.info(f"**Current:** {CURRENT_HARDWARE}")

    if baseline_hardware:
        st.sidebar.markdown(f"**Available Baselines:**")
        for hw in baseline_hardware:
            st.sidebar.caption(f"• {hw}")

    # Page: Dashboard
    if page == "🏠 Dashboard":
        st.markdown('<h1 class="main-header">Dashboard</h1>', unsafe_allow_html=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Experiments", len(experiments))
        with col2:
            st.metric("Batch Sizes", len(batches))
        with col3:
            st.metric("Models", len(MODEL_COLUMNS))
        with col4:
            st.metric("Baseline Hardware", len(baseline_hardware))

        st.markdown("---")

        # Experiment selector
        selected_exp = st.selectbox("Select Experiment", experiments)

        # Filter data for selected experiment
        exp_data = csv_df[csv_df['experiment'] == selected_exp].copy()

        # Display results table with baseline comparison
        st.subheader(f"Results: {selected_exp}")

        # Prepare display data
        display_cols = ['batch', 'M', 'N', 'K']
        for model_key, model_info in MODEL_COLUMNS.items():
            display_cols.append(f"{model_key}_gflops")
            display_cols.append(model_info["latency"])
            if model_info["vs_baseline"] in exp_data.columns:
                display_cols.append(model_info["vs_baseline"])
        if 'baseline_tflops' in exp_data.columns:
            display_cols.append('baseline_tflops')

        display_df = exp_data[display_cols].copy()

        # Rename columns for display
        column_rename = {
            'batch': 'Batch',
            'M': 'M',
            'N': 'N',
            'K': 'K',
            'baseline_tflops': '🎯 Baseline (TFLOPS)'
        }
        for model_key, model_info in MODEL_COLUMNS.items():
            column_rename[f"{model_key}_gflops"] = f"⚡ {model_info['name']} (GFLOPS)"
            column_rename[model_info["latency"]] = f"⏱️ {model_info['name']} (ms)"
            column_rename[model_info["vs_baseline"]] = f"📊 {model_info['name']} vs Baseline"
        display_df = display_df.rename(columns=column_rename)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Chart: GFLOPS comparison
        st.subheader("Performance Comparison (GFLOPS)")
        chart_data = exp_data[['M', 'batch'] + [f"{k}_gflops" for k in MODEL_COLUMNS.keys()]].copy()
        chart_data['batch'] = chart_data['batch'].str.replace('batch_', 'M=')

        fig = px.bar(
            chart_data,
            x='batch',
            y=[f"{k}_gflops" for k in MODEL_COLUMNS.keys()],
            title=f"GFLOPS by Batch Size - {selected_exp}",
            labels={'value': 'GFLOPS', 'variable': 'Model', 'batch': 'Batch Size'},
            barmode='group',
            color_discrete_map={f"{k}_gflops": v["color"] for k, v in MODEL_COLUMNS.items()}
        )
        fig.update_layout(
            xaxis_title="Batch Size",
            yaxis_title="GFLOPS",
            legend_title="Model"
        )
        new_names = {f"{k}_gflops": v["name"] for k, v in MODEL_COLUMNS.items()}
        fig.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))
        st.plotly_chart(fig, use_container_width=True)

        # Chart: Baseline comparison percentage
        if any(model_info["vs_baseline"] in exp_data.columns for model_info in MODEL_COLUMNS.values()):
            st.subheader("Baseline Comparison (%)")
            pct_data = exp_data[['M', 'batch']].copy()
            pct_data['batch'] = pct_data['batch'].str.replace('batch_', 'M=')

            for model_key, model_info in MODEL_COLUMNS.items():
                if model_info["vs_baseline"] in exp_data.columns:
                    pct_data[f"{model_key}_pct"] = exp_data[model_info["vs_baseline"]].apply(parse_percentage)

            pct_cols_list = [f"{k}_pct" for k in MODEL_COLUMNS.keys() if f"{k}_pct" in pct_data.columns]

            fig_pct = px.bar(
                pct_data,
                x='batch',
                y=pct_cols_list,
                title=f"Percentage of Baseline Performance - {selected_exp}",
                labels={'value': '% of Baseline', 'variable': 'Model', 'batch': 'Batch Size'},
                barmode='group',
                color_discrete_map={f"{k}_pct": v["color"] for k, v in MODEL_COLUMNS.items()}
            )
            fig_pct.update_layout(
                xaxis_title="Batch Size",
                yaxis_title="% of Baseline",
                legend_title="Model"
            )
            fig_pct.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Baseline")
            new_names_pct = {f"{k}_pct": v["name"] for k, v in MODEL_COLUMNS.items()}
            fig_pct.for_each_trace(lambda t: t.update(name=new_names_pct.get(t.name, t.name)))
            st.plotly_chart(fig_pct, use_container_width=True)

    # Page: Model Comparison
    elif page == "📊 Model Comparison":
        st.markdown('<h1 class="main-header">Model Comparison</h1>', unsafe_allow_html=True)

        # Filters
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            selected_exp = st.selectbox("Experiment", experiments, key="comp_exp")
        with col2:
            batch_filter = st.multiselect("Batch Size (M)", batches, default=batches)
        with col3:
            model_filter = st.multiselect("Models", list(MODEL_COLUMNS.keys()),
                                        default=list(MODEL_COLUMNS.keys()), key="comp_model")

        # Filter data
        filtered_df = csv_df[
            (csv_df['experiment'] == selected_exp) &
            (csv_df['M'].isin(batch_filter))
        ].copy()

        if filtered_df.empty:
            st.warning("No data matching the filters.")
            return

        # Comparison table
        st.subheader("GFLOPS Comparison with Baseline")

        comp_data = []
        for _, row in filtered_df.iterrows():
            comp_row = {
                'Batch': row['batch'],
                'M': row['M'],
                'N': row['N'],
                'K': row['K']
            }
            for model_key in model_filter:
                model_info = MODEL_COLUMNS[model_key]
                gflops_col = f"{model_key}_gflops"
                if gflops_col in filtered_df.columns:
                    comp_row[f"{model_info['name']} (GFLOPS)"] = row[gflops_col]
                if model_info["vs_baseline"] in filtered_df.columns:
                    comp_row[f"{model_info['name']} vs Baseline"] = row[model_info["vs_baseline"]]

            if 'baseline_tflops' in filtered_df.columns:
                comp_row['Baseline (TFLOPS)'] = row['baseline_tflops']
            comp_data.append(comp_row)

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Chart
        st.subheader("Visual Comparison")
        chart_data = filtered_df[['M', 'batch'] + [f"{k}_gflops" for k in model_filter]].copy()
        chart_data['batch'] = chart_data['batch'].str.replace('batch_', 'M=')

        fig = px.bar(
            chart_data,
            x='batch',
            y=[f"{k}_gflops" for k in model_filter],
            title=f"Model Comparison - {selected_exp}",
            labels={'value': 'GFLOPS', 'variable': 'Model', 'batch': 'Batch Size'},
            barmode='group',
            color_discrete_map={f"{k}_gflops": MODEL_COLUMNS[k]["color"] for k in model_filter}
        )
        new_names = {f"{k}_gflops": v["name"] for k, v in MODEL_COLUMNS.items() if k in model_filter}
        fig.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))
        st.plotly_chart(fig, use_container_width=True)

    # Page: Leaderboard
    elif page == "🏆 Leaderboard":
        st.markdown('<h1 class="main-header">Performance Leaderboard</h1>', unsafe_allow_html=True)

        # Filter by batch size
        selected_batch = st.selectbox("Select Batch Size", ["All"] + [f"M={b}" for b in batches])

        leaderboard_data = []
        for _, row in csv_df.iterrows():
            if selected_batch != "All" and f"M={row['M']}" != selected_batch:
                continue

            for model_key, model_info in MODEL_COLUMNS.items():
                gflops_col = f"{model_key}_gflops"
                if gflops_col in csv_df.columns:
                    leaderboard_data.append({
                        'Experiment': row['experiment'],
                        'Batch': row['batch'],
                        'M': row['M'],
                        'N': row['N'],
                        'K': row['K'],
                        'Model': model_info['name'],
                        'GFLOPS': row[gflops_col],
                        'Latency (ms)': row[model_info["latency"]],
                        'TFLOPS': row[model_info["tflops"]],
                        'vs Baseline': row.get(model_info["vs_baseline"], "N/A")
                    })

        leaderboard_df = pd.DataFrame(leaderboard_data)
        leaderboard_df = leaderboard_df.sort_values('GFLOPS', ascending=False)

        st.dataframe(leaderboard_df.head(50), use_container_width=True, hide_index=True)

        # Top performers chart
        st.subheader("Top 20 Performers")
        top_20 = leaderboard_df.head(20)
        fig = px.bar(
            top_20,
            x='GFLOPS',
            y='Model',
            color='Experiment',
            orientation='h',
            title="Top 20 GFLOPS Performance"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Page: Baseline Data
    elif page == "🖥️ Baseline Data":
        st.markdown('<h1 class="main-header">Baseline Performance Data</h1>', unsafe_allow_html=True)

        if not baseline_data:
            st.warning("No baseline data available.")
            return

        st.info(f"Baseline data source: {BASELINE_FILE}")

        # Hardware selector
        selected_hw = st.selectbox("Select Hardware", baseline_hardware)

        # Get all unique experiment types from baseline
        exp_types = set()
        for key in baseline_data.keys():
            parts = key.split('_')
            if len(parts) >= 3:
                exp_types.add('_'.join(parts[0:3]))  # e.g., w4a32c8_q4_0

        exp_type = st.selectbox("Experiment Type", sorted(exp_types))

        # Filter and display baseline data
        baseline_display = []
        for key, entry in baseline_data.items():
            if not key.startswith(exp_type):
                continue

            hw_data = entry.get('hardware', {}).get(selected_hw)
            if hw_data:
                # Parse dimensions from key
                # Baseline format: m=K_input, n=M_batch, k=N_output
                parts = key.split('_')
                m_val = n_val = k_val = None
                for part in parts:
                    if part.startswith('m'):
                        m_val = int(part[1:])  # This is K (input features)
                    elif part.startswith('n'):
                        n_val = int(part[1:])  # This is M (batch size)
                    elif part.startswith('k'):
                        k_val = int(part[1:])  # This is N (output features)

                baseline_display.append({
                    'Key': key,
                    'Batch (M)': n_val,      # n = batch size
                    'Output (N)': k_val,     # k = output features
                    'Input (K)': m_val,      # m = input features
                    'GFLOPS': hw_data.get('gflops', 0),
                    'TFLOPS': hw_data.get('tflops', 0),
                    'Latency (ms)': hw_data.get('latency_ms', 0)
                })

        baseline_df = pd.DataFrame(baseline_display)
        if not baseline_df.empty:
            st.dataframe(baseline_df.sort_values('GFLOPS', ascending=False), use_container_width=True, hide_index=True)

            # Chart
            st.subheader(f"Baseline Performance - {selected_hw}")
            fig = px.scatter(
                baseline_df,
                x='Batch (M)',
                y='GFLOPS',
                color='Output (N)',
                size='Input (K)',
                title=f"Baseline GFLOPS vs Batch Size (M) - {selected_hw}",
                labels={'Batch (M)': 'Batch Size (M)', 'GFLOPS': 'GFLOPS', 'Output (N)': 'Output Features (N)', 'Input (K)': 'Input Features (K)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No baseline data found for {exp_type} on {selected_hw}")

        # Show all hardware comparison
        st.subheader("Hardware Comparison")
        if exp_type:
            hw_comparison = []
            for key, entry in baseline_data.items():
                if key.startswith(exp_type):
                    row = {'Key': key}
                    for hw in baseline_hardware:
                        hw_data = entry.get('hardware', {}).get(hw, {})
                        row[f"{hw} (GFLOPS)"] = hw_data.get('gflops', 0)
                    hw_comparison.append(row)

            hw_df = pd.DataFrame(hw_comparison)
            if not hw_df.empty:
                st.dataframe(hw_df.head(20), use_container_width=True, hide_index=True)

    # Page: Detailed Analysis
    elif page == "📈 Detailed Analysis":
        st.markdown('<h1 class="main-header">Detailed Analysis</h1>', unsafe_allow_html=True)

        # Selectors
        col1, col2 = st.columns([2, 2])
        with col1:
            selected_exp = st.selectbox("Experiment", experiments, key="detail_exp")
        with col2:
            selected_model = st.selectbox("Model", list(MODEL_COLUMNS.keys()), key="detail_model")

        model_info = MODEL_COLUMNS[selected_model]

        # Filter data
        exp_data = csv_df[csv_df['experiment'] == selected_exp].copy()

        if exp_data.empty:
            st.warning("No data for this experiment.")
            return

        # Metrics display
        st.subheader(f"{model_info['name']} Performance - {selected_exp}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_gflops = exp_data[f"{selected_model}_gflops"].mean()
            st.metric("Avg GFLOPS", f"{avg_gflops:.1f}")
        with col2:
            max_gflops = exp_data[f"{selected_model}_gflops"].max()
            st.metric("Max GFLOPS", f"{max_gflops:.1f}")
        with col3:
            avg_latency = exp_data[model_info["latency"]].mean()
            st.metric("Avg Latency (ms)", f"{avg_latency:.3f}")
        with col4:
            if model_info["vs_baseline"] in exp_data.columns:
                avg_vs_baseline = exp_data[model_info["vs_baseline"]].apply(parse_percentage).mean()
                st.metric("Avg vs Baseline", f"{avg_vs_baseline:.1f}%")

        # Detailed table with baseline
        detail_data = []
        for _, row in exp_data.iterrows():
            detail_row = {
                'Batch': row['batch'],
                'M': row['M'],
                'N': row['N'],
                'K': row['K'],
                'GFLOPS': row[f"{selected_model}_gflops"],
                'Latency (ms)': row[model_info["latency"]],
                'TFLOPS': row[model_info["tflops"]]
            }

            if 'baseline_tflops' in exp_data.columns:
                detail_row['Baseline TFLOPS'] = row['baseline_tflops']

            if model_info["vs_baseline"] in exp_data.columns:
                detail_row['vs Baseline'] = row[model_info["vs_baseline"]]

            detail_data.append(detail_row)

        detail_df = pd.DataFrame(detail_data)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        # Performance trend chart
        st.subheader("Performance Trend")
        fig = px.line(
            exp_data,
            x='M',
            y=f"{selected_model}_gflops",
            title=f"GFLOPS vs Batch Size - {model_info['name']}",
            markers=True,
            labels={'M': 'Batch Size (M)', f"{selected_model}_gflops": 'GFLOPS'}
        )
        fig.update_traces(line_color=model_info["color"])
        st.plotly_chart(fig, use_container_width=True)

        # Baseline trend
        if 'baseline_tflops' in exp_data.columns:
            st.subheader("Baseline Comparison Trend")
            trend_data = exp_data[['M', 'batch']].copy()
            trend_data['Model TFLOPS'] = exp_data[model_info["tflops"]]
            trend_data['Baseline TFLOPS'] = exp_data['baseline_tflops']
            trend_data['batch'] = trend_data['batch'].str.replace('batch_', 'M=')

            fig_trend = px.line(
                trend_data,
                x='M',
                y=['Model TFLOPS', 'Baseline TFLOPS'],
                title=f"TFLOPS Trend: Model vs Baseline - {model_info['name']}",
                markers=True,
                labels={'M': 'Batch Size (M)', 'value': 'TFLOPS', 'variable': 'Source'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    **KernelEvalPlus** - LLM-driven CUDA kernel generation and testing.

    Inspired by [FlashInfer Bench](https://bench.flashinfer.ai)
    """)

if __name__ == "__main__":
    main()
