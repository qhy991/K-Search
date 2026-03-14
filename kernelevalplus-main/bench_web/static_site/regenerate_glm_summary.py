#!/usr/bin/env python3
"""
Regenerate glm-5-0212_summary.csv with all M values (M=1,2,3,4,5,8,512)
from glm-5-0212_results.csv

The original summary file only had M=1 and M=512, but the performance chart
needs all M values to render the bars correctly.
"""

import csv
import re
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
RESULTS_FILE = DATA_DIR / "glm-5-0212_results.csv"
SUMMARY_FILE = DATA_DIR / "glm-5-0212_summary.csv"

# M values we need
M_VALUES = ['1', '2', '3', '4', '5', '8', '512']

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
            # Fallback: use name parsing
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
        # Clean up suffixes
        for suffix in ['_final', '_v1', '_v2', '_v3', '_v4', '_v5', '_v6', '_v7', '_v8', '_v9',
                       '_v10', '_v11', '_v12', '_optimization', '_experiment', '_20260213',
                       '_20260214', '_154842', '_001', '_005', '_007', '_009']:
            task = task.replace(suffix, '')
        task = task.rstrip('_')

    # Build the experiment name
    # Format: w{weight_bits}a{act_bits}c{accum_bits}_{quant}_fp32_int8_{task}_n{n}_k{k}_best
    if variant == 'Q4_0':
        prefix = 'w4a32c8_q4_0'
    elif variant == 'Q4_1':
        prefix = 'w4a32c8_q4_1'
    elif variant == 'Q8_0':
        prefix = 'w8a32c8_q8_0'
    else:
        prefix = variant.lower().replace('_', '')

    # Construct name
    exp_name = f'{prefix}_fp32_int8_{task}_n{n_val}_k{k_val}_best'
    return exp_name

def get_task_from_name(name: str) -> str:
    """Extract task name from experiment name."""
    # Remove variant prefix and suffix
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

    # Normalize task names to match expected format
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

def main():
    print("Reading results file...")

    # Read all results
    results = []
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    print(f"Found {len(results)} result rows")

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

    print(f"Grouped into {len(grouped)} task/variant combinations")

    # For each group, find the "best" entry (highest M=512 GFLOPS)
    best_entries = []
    for (task, variant), group in grouped.items():
        # Find row with max M=512 GFLOPS
        best_row = max(group['rows'], key=lambda r: float(r.get('M=512_gflops', 0) or 0))

        # Create summary entry with proper experiment name
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

        # Calculate small_avg (average of M=1 to M=8)
        m1_gflops = float(best_row.get('M=1_gflops', 0) or 0)
        m2_gflops = float(best_row.get('M=2_gflops', 0) or 0)
        m3_gflops = float(best_row.get('M=3_gflops', 0) or 0)
        m4_gflops = float(best_row.get('M=4_gflops', 0) or 0)
        m5_gflops = float(best_row.get('M=5_gflops', 0) or 0)
        m8_gflops = float(best_row.get('M=8_gflops', 0) or 0)
        small_vals = [v for v in [m1_gflops, m2_gflops, m3_gflops, m4_gflops, m5_gflops, m8_gflops] if v > 0]
        small_avg = sum(small_vals) / len(small_vals) if small_vals else 0
        summary_entry['small_avg'] = f"{small_avg:.1f}"

        # Calculate scaling ratio
        m512_gflops = float(best_row.get('M=512_gflops', 0) or 0)
        scaling_ratio = m512_gflops / m1_gflops if m1_gflops > 0 else 0
        summary_entry['scaling_ratio'] = f"{scaling_ratio:.1f}"

        # Generate notes
        summary_entry['notes'] = f"{variant.lower()}_{task}"

        best_entries.append(summary_entry)

    # Sort by experiment_name and remove duplicates (keep best by M=512 GFLOPS)
    best_entries.sort(key=lambda x: (x['experiment_name'], -float(x['M=512_gflops'] or 0)))

    # Deduplicate by experiment_name, keeping the first (highest M=512)
    seen = {}
    unique_entries = []
    for e in best_entries:
        exp_name = e['experiment_name']
        if exp_name not in seen:
            seen[exp_name] = True
            unique_entries.append(e)

    best_entries = unique_entries
    best_entries.sort(key=lambda x: x['experiment_name'])

    print(f"After deduplication: {len(best_entries)} unique entries")

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

    # Write summary file
    print(f"Writing summary file with {len(best_entries)} entries...")

    # Build column headers
    columns = ['experiment_name', 'variant', 'dimensions', 'best_attempt']
    for m in M_VALUES:
        columns.extend([f'M={m}_gflops', f'M={m}_time_ms', f'M={m}_baseline_pct'])
    columns.extend(['small_avg', 'scaling_ratio', 'rank_m1', 'rank_m512', 'notes'])

    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(best_entries)

    print(f"✓ Created {SUMMARY_FILE}")
    print(f"  - {len(best_entries)} entries")
    print(f"  - M values: {', '.join(M_VALUES)}")

    # Show sample entries
    print("\nSample entries:")
    for e in best_entries[:5]:
        print(f"  - {e['experiment_name']}")

if __name__ == '__main__':
    main()
