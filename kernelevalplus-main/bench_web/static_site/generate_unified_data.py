#!/usr/bin/env python3
"""
Generate unified performance data file for the benchmark website.

Consolidates all model performance data into a single CSV file.
"""

import csv
from pathlib import Path

# Paths
STATIC_DIR = Path(__file__).parent
DATA_DIR = STATIC_DIR / "data"

# Source files
DS2_SUMMARY = DATA_DIR / "deepseek_v2_best_summary.csv"
GLM_RESULTS = DATA_DIR / "glm-5-0212_results.csv"

# All M values
M_VALUES = ['1', '2', '3', '4', '5', '8', '512']

def parse_experiment_name(exp_name):
    """Parse experiment name to extract model, task, and variant."""
    exp_name_lower = exp_name.lower()

    # Extract variant
    if '_q4_0_' in exp_name_lower or exp_name_lower.startswith('q4_0'):
        variant = 'Q4_0'
    elif '_q4_1_' in exp_name_lower or exp_name_lower.startswith('q4_1'):
        variant = 'Q4_1'
    elif '_q8_0_' in exp_name_lower or exp_name_lower.startswith('q8_0'):
        variant = 'Q8_0'
    else:
        variant = 'Unknown'

    # Extract model and task
    model = 'unknown'
    task = 'unknown'
    dimensions = 'unknown'

    # Special cases for GLM experiments
    if 'q4_0_experiments' in exp_name_lower:
        model = 'deepseek_v3'
        task = 'att_out'
        dimensions = '7168x7168'
    elif 'q8_0_experiments' in exp_name_lower:
        model = 'deepseek_v3'
        task = 'att_out'
        dimensions = '7168x7168'
    elif 'deepseek_v3_att_out' in exp_name_lower:
        model = 'deepseek_v3'
        task = 'att_out'
        dimensions = '7168x7168'
    elif 'llama3_8b_att_out' in exp_name_lower:
        model = 'llama3_8b'
        task = 'att_out'
        dimensions = '4096x4096'
    elif 'qwen2_5_7b_att_out' in exp_name_lower:
        model = 'qwen2_5_7b'
        task = 'att_out'
        dimensions = '3584x3584'
    elif 'qwen3_4b_att_out' in exp_name_lower:
        model = 'qwen3_4b'
        task = 'att_out'
        dimensions = '2560x2560'
    elif 'mixtral8x7b_moe_up' in exp_name_lower:
        model = 'mixtral8x7b'
        task = 'moe_up'
        dimensions = '14336x4096'
    elif 'mixtral_moe_up' in exp_name_lower:
        model = 'mixtral8x7b'
        task = 'moe_up'
        dimensions = '14336x4096'
    elif 'qwen3_4b_optimization' in exp_name_lower:
        model = 'qwen3_4b'
        task = 'att_out'
        dimensions = '2560x2560'
    elif 'ds3_moe_routing_down' in exp_name_lower:
        model = 'deepseek_v3'
        task = 'moe_routing_down'
        dimensions = '2048x7168'
    # DeepSeek-V2 patterns
    elif 'ds2_moe_down' in exp_name_lower or 'deepseek_v2_moe_down' in exp_name_lower:
        model = 'deepseek_v2'
        task = 'moe_down'
    elif 'ds2_moe_routing_down' in exp_name_lower or 'deepseek_v2_moe_routing_down' in exp_name_lower:
        model = 'deepseek_v2'
        task = 'moe_routing_down'
    elif 'ds2_moe_routing_up' in exp_name_lower or 'deepseek_v2_moe_routing_up' in exp_name_lower:
        model = 'deepseek_v2'
        task = 'moe_routing_up'
    elif 'ds2_moe_up' in exp_name_lower or 'deepseek_v2_moe_up' in exp_name_lower:
        model = 'deepseek_v2'
        task = 'moe_up'
    elif 'deepseek_v2_att_out' in exp_name_lower:
        model = 'deepseek_v2'
        task = 'att_out'
    elif 'deepseek_v2_lm_head' in exp_name_lower:
        model = 'deepseek_v2'
        task = 'lm_head'
    else:
        # Try to parse from the name pattern
        if 'att_out' in exp_name_lower:
            task = 'att_out'
        elif 'lm_head' in exp_name_lower:
            task = 'lm_head'
        elif 'moe_up' in exp_name_lower:
            task = 'moe_up'
        elif 'moe_down' in exp_name_lower:
            task = 'moe_down'
        elif 'moe_routing_up' in exp_name_lower:
            task = 'moe_routing_down'
        elif 'moe_routing_down' in exp_name_lower:
            task = 'moe_routing_down'

        # Extract model
        if 'deepseek_v2' in exp_name_lower or 'ds2_' in exp_name_lower:
            model = 'deepseek_v2'
        elif 'deepseek_v3' in exp_name_lower or 'ds3_' in exp_name_lower:
            model = 'deepseek_v3'
        elif 'llama3_8b' in exp_name_lower:
            model = 'llama3_8b'
        elif 'mixtral8x7b' in exp_name_lower:
            model = 'mixtral8x7b'
        elif 'qwen2_5_7b' in exp_name_lower:
            model = 'qwen2_5_7b'
        elif 'qwen3_4b' in exp_name_lower:
            model = 'qwen3_4b'

    # Extract dimensions
    import re
    n_match = re.search(r'_n(\d+)', exp_name)
    k_match = re.search(r'_k(\d+)', exp_name)
    if n_match and k_match:
        dimensions = f"{n_match.group(1)}x{k_match.group(1)}"

    return model, task, variant, dimensions

def read_csv_safe(filepath):
    """Read CSV file safely."""
    entries = []
    if not filepath.exists():
        print(f"  Warning: {filepath.name} not found")
        return entries

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries

def main():
    print("=" * 60)
    print("Generating unified performance data")
    print("=" * 60)

    all_entries = []

    # Process DeepSeek-V2 summary
    print(f"Reading {DS2_SUMMARY.name}...")
    ds2_entries = read_csv_safe(DS2_SUMMARY)
    print(f"  - Found {len(ds2_entries)} entries")

    for entry in ds2_entries:
        exp_name = entry.get('experiment_name', '')
        model, task, variant, dimensions = parse_experiment_name(exp_name)

        unified_entry = {
            'experiment_name': exp_name,
            'model': model,
            'task': task,
            'variant': entry.get('variant', variant),
            'dimensions': dimensions,
            'best_attempt': entry.get('best_attempt', ''),
        }

        # Copy all M values
        for m in M_VALUES:
            for suffix in ['_gflops', '_time_ms', '_baseline_pct']:
                col = f'M={m}{suffix}'
                unified_entry[col] = entry.get(col, '')

        unified_entry['small_avg'] = entry.get('small_avg', '')
        unified_entry['scaling_ratio'] = entry.get('scaling_ratio', '')

        all_entries.append(unified_entry)

    # Process GLM results
    print(f"Reading {GLM_RESULTS.name}...")
    glm_entries = read_csv_safe(GLM_RESULTS)
    print(f"  - Found {len(glm_entries)} entries")

    # Group by (model, task, variant) to find best for each group
    grouped = {}
    for entry in glm_entries:
        exp_name = entry.get('experiment_name', '')
        model, task, variant, dimensions = parse_experiment_name(exp_name)

        key = (model, task, variant)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(entry)

    # For each group, find the entry with max M=512 GFLOPS
    for key, entries_list in grouped.items():
        best_entry = max(entries_list, key=lambda e: float(e.get('M=512_gflops', 0) or 0))

        model, task, variant, dimensions = parse_experiment_name(best_entry['experiment_name'])

        unified_entry = {
            'experiment_name': best_entry['experiment_name'],
            'model': model,
            'task': task,
            'variant': variant,
            'dimensions': dimensions,
            'best_attempt': best_entry['experiment_name'],
        }

        # Copy all M values
        for m in M_VALUES:
            for suffix in ['_gflops', '_time_ms', '_baseline_pct']:
                col = f'M={m}{suffix}'
                # GLM results use _latency_ms instead of _time_ms
                if suffix == '_time_ms':
                    col_alt = f'M={m}_latency_ms'
                    val = best_entry.get(col_alt, '')
                else:
                    val = best_entry.get(col, '')
                unified_entry[col] = val

        # Calculate small_avg and scaling_ratio
        vals = []
        for m in M_VALUES:
            v = float(best_entry.get(f'M={m}_gflops', 0) or 0)
            if v > 0 and m in ['1', '2', '3', '4', '5', '8']:
                vals.append(v)

        if vals:
            small_avg = sum(vals) / len(vals)
        else:
            small_avg = 0
        unified_entry['small_avg'] = f"{small_avg:.1f}"

        m1 = float(best_entry.get('M=1_gflops', 0) or 0)
        m512 = float(best_entry.get('M=512_gflops', 0) or 0)
        scaling = m512 / m1 if m1 > 0 else 0
        unified_entry['scaling_ratio'] = f"{scaling:.1f}"

        all_entries.append(unified_entry)

    # Calculate ranks
    m1_values = [(i, float(e.get('M=1_gflops', 0) or 0)) for i, e in enumerate(all_entries)]
    m1_values.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx, _) in enumerate(m1_values, 1):
        all_entries[idx]['rank_m1'] = rank

    m512_values = [(i, float(e.get('M=512_gflops', 0) or 0)) for i, e in enumerate(all_entries)]
    m512_values.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx, _) in enumerate(m512_values, 1):
        all_entries[idx]['rank_m512'] = rank

    # Write unified file
    output_file = DATA_DIR / "unified_performance_summary.csv"
    columns = ['experiment_name', 'model', 'task', 'variant', 'dimensions', 'best_attempt']
    for m in M_VALUES:
        columns.extend([f'M={m}_gflops', f'M={m}_time_ms', f'M={m}_baseline_pct'])
    columns.extend(['small_avg', 'scaling_ratio', 'rank_m1', 'rank_m512'])

    print(f"\nWriting unified summary to {output_file.name}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"  - {len(all_entries)} total entries")

    # Show model and task distribution
    models = {}
    tasks = {}
    for e in all_entries:
        models[e['model']] = models.get(e['model'], 0) + 1
        tasks[e['task']] = tasks.get(e['task'], 0) + 1

    print(f"  - Models: {dict(models)}")
    print(f"  - Tasks: {dict(tasks)}")

    print("\n" + "=" * 60)
    print("Files created:")
    print("  1. unified_performance_summary.csv - All performance data")
    print("  2. baseline.json - Baseline reference data")
    print("=" * 60)

if __name__ == '__main__':
    main()
