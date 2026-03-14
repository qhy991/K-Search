#!/usr/bin/env python3
"""
Clean implementation hints from definition files (Phase 3).
"""

import json
import os
import glob
import sys
import re

DEFINITIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'definitions', 'quant_gemm')

def clean_phase3(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    changes = []
    
    if 'formula' in data and 'compute_flow' in data['formula']:
        old_flow = data['formula']['compute_flow']
        # "2. INT8 GEMM with BLOCK_Q4_0 weights"
        # -> "2. GEMM with BLOCK_Q4_0 weights"
        new_flow = old_flow.replace('INT8 GEMM', 'GEMM').replace('INT8 ', '')
        if new_flow != old_flow:
            data['formula']['compute_flow'] = new_flow
            changes.append(f"  Cleaned INT8 from compute_flow")
            
    if changes:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
        return changes
    return None

def main():
    json_files = glob.glob(os.path.join(DEFINITIONS_DIR, '**', '*.json'), recursive=True)
    for filepath in sorted(json_files):
        relpath = os.path.relpath(filepath, DEFINITIONS_DIR)
        changes = clean_phase3(filepath)
        if changes:
            print(f"✅ {relpath}")
            for c in changes:
                print(c)

if __name__ == '__main__':
    main()
