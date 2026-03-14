#!/usr/bin/env python3
"""
Generate static data files for the benchmark website.
"""

import csv
import json
import re
import shutil
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATIC_DIR = Path(__file__).parent
DATA_DIR = STATIC_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# M values for performance chart
M_VALUES = ['1', '2', '3', '4', '5', '8', '512']


def regenerate_glm_summary(results_file: Path, summary_file: Path):
    """
    Regenerate glm-5-0212_summary.csv with all M values (M=1,2,3,4,5,8,512)
    from glm-5-0212_results.csv.

    The original summary file only had M=1 and M=512, but the performance chart
    needs all M values to render the bars correctly.
    """

    def normalize_experiment_name(name: str, variant: str, dimensions: str) -> str:
        """Convert experiment name to the standardized summary format."""

        # Handle special case entries first
        if 'q4_0_experiments' in name:
            return f'w4a32c8_q4_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_best'
        if 'q8_0_experiments' in name:
            return f'w8a32c8_q8_0_fp32_int8_deepseek_v3_att_out_n7168_k7168_best'

        # Parse dimensions to get n and k values
        n_match = re.search(r'_n(\d+)', name)
        k_match = re.search(r'_k(\d+)', name)

        if not n_match or not k_match:
            # If we can't find n/k in the name, derive from dimensions
            dims = dimensions.split('x')
            if len(dims) == 2:
                n_val = dims[0]
                k_val = dims[1]
            else:
                n_val = 'unknown'
                k_val = 'unknown'
        else:
            n_val = n_match.group(1)
            k_val = k_match.group(1)

        # Determine the base task name
        task_patterns = {
            'deepseek_v2_att_out': 'deepseek_v2_att_out',
            'deepseek_v3_att_out': 'deepseek_v3_att_out',
            'ds3_moe_routing_down': 'ds3_moe_routing_down',
            'ds3_moe_routing_up': 'ds3_moe_routing_up',
            'ds3_moe_down': 'ds3_moe_down',
            'ds3_moe_up': 'ds3_moe_up',
            'llama3_8b_att_out': 'llama3_8b_att_out',
            'mixtral8x7b_moe_up': 'mixtral8x7b_moe_up',
            'mixtral_moe_up': 'mixtral8x7b_moe_up',
            'qwen2_5_7b_att_out': 'qwen2_5_7b_att_out',
            'qwen3_4b_att_out': 'qwen3_4b_att_out',
        }

        task = None
        for pattern, task_name in task_patterns.items():
            if pattern in name:
                task = task_name
                break

        if task is None:
            # Try to extract task from name
            task = name.replace('w4a32c8_q4_0_fp32_int8_', '')
            task = task.replace('w4a32c8_q4_1_fp32_int8_', '')
            task = task.replace('w8a32c8_q8_0_fp32_int8_', '')
            for suffix in ['_final', '_v1', '_v2', '_v3', '_v4', '_v5', '_v6', '_v7', '_v8', '_v9',
                           '_v10', '_v11', '_v12', '_optimization', '_experiment', '_20260213',
                           '_20260214', '_154842', '_001', '_005', '_007', '_009']:
                task = task.replace(suffix, '')
            task = task.rstrip('_')

        # Build the experiment name
        if variant == 'Q4_0':
            prefix = 'w4a32c8_q4_0'
        elif variant == 'Q4_1':
            prefix = 'w4a32c8_q4_1'
        elif variant == 'Q8_0':
            prefix = 'w8a32c8_q8_0'
        else:
            prefix = variant.lower().replace('_', '')

        exp_name = f'{prefix}_fp32_int8_{task}_n{n_val}_k{k_val}_best'
        return exp_name

    def get_task_from_name(name: str) -> str:
        """Extract task name from experiment name."""
        name = name.replace('w4a32c8_q4_0_fp32_int8_', '')
        name = name.replace('w4a32c8_q4_1_fp32_int8_', '')
        name = name.replace('w8a32c8_q8_0_fp32_int8_', '')
        name = name.replace('_best', '')
        name = name.replace('_final', '')
        name = name.replace('_v1', '').replace('_v2', '').replace('_v3', '')
        name = name.replace('_v4', '').replace('_v5', '').replace('_v6', '')
        name = name.replace('_v7', '').replace('_v8', '').replace('_v9', '')
        name = name.replace('_v10', '').replace('_v11', '').replace('_v12', '')
        name = name.replace('_optimization', '').replace('_experiment', '')
        name = name.replace('_20260213', '').replace('_20260214', '').replace('_154842', '')
        name = name.replace('_001', '').replace('_005', '').replace('_007', '').replace('_009', '')

        task_mapping = {
            'deepseek_v2_att_out_n5120_k5120': 'deepseek_v2_att_out',
            'deepseek_v3_att_out_n7168_k7168': 'deepseek_v3_att_out',
            'ds3_moe_routing_down_n2048_k7168': 'ds3_moe_routing_down',
            'llama3_8b_att_out_n4096_k4096': 'llama3_8b_att_out',
            'mixtral8x7b_moe_up_n14336_k4096': 'mixtral8x7b_moe_up',
            'qwen2_5_7b_att_out_n3584_k3584': 'qwen2_5_7b_att_out',
            'qwen3_4b_att_out_n2560_k2560': 'qwen3_4b_att_out',
        }

        for key, value in task_mapping.items():
            if key in name:
                return value

        return name

    # Read all results
    results = []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    # Group by task and variant
    grouped = {}
    for row in results:
        task = get_task_from_name(row['experiment_name'])
        variant = row['variant']

        key = (task, variant)
        if key not in grouped:
            grouped[key] = {
                'task': task,
                'variant': variant,
                'dimensions': row['dimensions'],
                'rows': []
            }
        grouped[key]['rows'].append(row)

    # For each group, find the "best" entry (highest M=512 GFLOPS)
    best_entries = []
    for (task, variant), group in grouped.items():
        best_row = max(group['rows'], key=lambda r: float(r.get('M=512_gflops', 0) or 0))

        exp_name = normalize_experiment_name(best_row['experiment_name'], variant, group['dimensions'])

        summary_entry = {
            'experiment_name': exp_name,
            'variant': variant,
            'dimensions': group['dimensions'],
            'best_attempt': best_row['experiment_name'],
        }

        # Add all M values
        for m in M_VALUES:
            summary_entry[f'M={m}_gflops'] = best_row.get(f'M={m}_gflops', '')
            summary_entry[f'M={m}_time_ms'] = best_row.get(f'M={m}_latency_ms', '')
            summary_entry[f'M={m}_baseline_pct'] = best_row.get(f'M={m}_baseline_pct', '')

        # Calculate small_avg and scaling_ratio
        m1_gflops = float(best_row.get('M=1_gflops', 0) or 0)
        m2_gflops = float(best_row.get('M=2_gflops', 0) or 0)
        m3_gflops = float(best_row.get('M=3_gflops', 0) or 0)
        m4_gflops = float(best_row.get('M=4_gflops', 0) or 0)
        m5_gflops = float(best_row.get('M=5_gflops', 0) or 0)
        m8_gflops = float(best_row.get('M=8_gflops', 0) or 0)
        small_vals = [v for v in [m1_gflops, m2_gflops, m3_gflops, m4_gflops, m5_gflops, m8_gflops] if v > 0]
        small_avg = sum(small_vals) / len(small_vals) if small_vals else 0
        summary_entry['small_avg'] = f"{small_avg:.1f}"

        m512_gflops = float(best_row.get('M=512_gflops', 0) or 0)
        scaling_ratio = m512_gflops / m1_gflops if m1_gflops > 0 else 0
        summary_entry['scaling_ratio'] = f"{scaling_ratio:.1f}"

        summary_entry['notes'] = f"{variant.lower()}_{task}"
        best_entries.append(summary_entry)

    # Sort and deduplicate
    best_entries.sort(key=lambda x: (x['experiment_name'], -float(x['M=512_gflops'] or 0)))
    seen = {}
    unique_entries = []
    for e in best_entries:
        if e['experiment_name'] not in seen:
            seen[e['experiment_name']] = True
            unique_entries.append(e)

    best_entries = unique_entries
    best_entries.sort(key=lambda x: x['experiment_name'])

    # Calculate ranks
    m1_values = [(e['experiment_name'], float(e['M=1_gflops'] or 0)) for e in best_entries]
    m1_values.sort(key=lambda x: x[1], reverse=True)
    for rank, (exp_name, _) in enumerate(m1_values, 1):
        for e in best_entries:
            if e['experiment_name'] == exp_name:
                e['rank_m1'] = rank
                break

    m512_values = [(e['experiment_name'], float(e['M=512_gflops'] or 0)) for e in best_entries]
    m512_values.sort(key=lambda x: x[1], reverse=True)
    for rank, (exp_name, _) in enumerate(m512_values, 1):
        for e in best_entries:
            if e['experiment_name'] == exp_name:
                e['rank_m512'] = rank
                break

    # Build column headers
    columns = ['experiment_name', 'variant', 'dimensions', 'best_attempt']
    for m in M_VALUES:
        columns.extend([f'M={m}_gflops', f'M={m}_time_ms', f'M={m}_baseline_pct'])
    columns.extend(['small_avg', 'scaling_ratio', 'rank_m1', 'rank_m512', 'notes'])

    # Write summary file
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(best_entries)

    print(f"✓ Generated glm-5-0212_summary.csv with {len(best_entries)} entries")

# Source files
CSV_SOURCE = PROJECT_ROOT / "data" / "experiments" / "results.csv"
BASELINE_SOURCE = PROJECT_ROOT / "data" / "baseline" / "baseline_data_compact.json"
DS2_BEST_SOURCE = PROJECT_ROOT / "batch_test_deepseek_v2_best" / "deepseek_v2_best_summary_extended.csv"

print(f"Static site directory: {STATIC_DIR}")
print(f"Data directory: {DATA_DIR}")

# Copy CSV
if CSV_SOURCE.exists():
    shutil.copy(CSV_SOURCE, DATA_DIR / "experiments.csv")
    print(f"✓ Copied experiments.csv")
else:
    print(f"✗ CSV source not found: {CSV_SOURCE}")

# Copy and process baseline JSON
if BASELINE_SOURCE.exists():
    with open(BASELINE_SOURCE) as f:
        baseline = json.load(f)

    with open(DATA_DIR / "baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"✓ Copied baseline.json with {len(baseline)} entries")
else:
    print(f"✗ Baseline source not found: {BASELINE_SOURCE}")

# Copy new operators baseline JSON (separate files)
NEW_OPERATORS_FILES = [
    ("flash_attn_baseline.json", "Flash Attention"),
    ("rms_norm_baseline.json", "RMS Norm"),
    ("topk_baseline.json", "TopK")
]

for filename, display_name in NEW_OPERATORS_FILES:
    source_file = PROJECT_ROOT / "data" / "baseline" / filename
    if source_file.exists():
        with open(source_file) as f:
            data = json.load(f)
        with open(DATA_DIR / filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Copied {filename} ({len(data)} entries)")
    else:
        print(f"⚠ {filename} not found")

# Copy DeepSeek-V2 best summary CSV
if DS2_BEST_SOURCE.exists():
    shutil.copy(DS2_BEST_SOURCE, DATA_DIR / "deepseek_v2_best_summary.csv")
    print(f"✓ Copied deepseek_v2_best_summary.csv")
else:
    print(f"✗ DeepSeek-V2 best summary not found: {DS2_BEST_SOURCE}")

# Generate GLM-5-0212 summary CSV with all M values
GLM_RESULTS_SOURCE = PROJECT_ROOT / "bench_web" / "static_site" / "data" / "glm-5-0212_results.csv"
if GLM_RESULTS_SOURCE.exists():
    try:
        regenerate_glm_summary(GLM_RESULTS_SOURCE, DATA_DIR / "glm-5-0212_summary.csv")
    except Exception as e:
        print(f"⚠ GLM summary generation skipped: {e}")
else:
    print(f"⚠ GLM results not found: {GLM_RESULTS_SOURCE}")

# Create GitHub Pages deployment info
print("\n" + "="*60)
print("Static site generated successfully!")
print("="*60)
print(f"\nFiles created in {DATA_DIR}:")
for f in DATA_DIR.glob("*"):
    print(f"  - {f.name}")
print(f"\nHTML file: {STATIC_DIR / 'index.html'}")
print("\nTo deploy to GitHub Pages:")
print("1. Create a gh-pages branch")
print("2. Copy the static_site directory contents")
print("3. Push to GitHub")
print("4. Enable GitHub Pages in repository settings")
