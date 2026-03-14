#!/usr/bin/env python3
"""
Generate kernels summary from definitions directory.

This script scans the definitions/ directory and generates a summary JSON file
containing all kernel definitions organized by operator type and model.
"""

import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFINITIONS_DIR = PROJECT_ROOT / "definitions"
OUTPUT_FILE = SCRIPT_DIR.parent / "data" / "kernels_summary.json"

def load_json_file(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def scan_definitions():
    """Scan definitions directory and build summary."""
    summary = {
        "operators": {},
        "models": {},
        "total_kernels": 0
    }

    # Scan each operator type directory
    for op_dir in DEFINITIONS_DIR.iterdir():
        if not op_dir.is_dir() or op_dir.name in ['docs']:
            continue

        op_type = op_dir.name
        summary["operators"][op_type] = {
            "display_name": op_type.replace("_", " ").title(),
            "kernels": [],
            "models": set(),
            "total_count": 0
        }

        # Scan each model directory
        for model_dir in op_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            summary["operators"][op_type]["models"].add(model_name)

            # Add to models section if not exists
            if model_name not in summary["models"]:
                summary["models"][model_name] = {
                    "display_name": format_model_name(model_name),
                    "operators": {},
                    "total_kernels": 0
                }

            # Add operator to model
            summary["models"][model_name]["operators"][op_type] = []

            # Scan JSON files
            for json_file in model_dir.glob("*.json"):
                kernel_def = load_json_file(json_file)
                if kernel_def:
                    # Add to operator section
                    kernel_info = {
                        "name": kernel_def.get("name", json_file.stem),
                        "variant": kernel_def.get("variant", "Unknown"),
                        "description": kernel_def.get("description", ""),
                        "model": model_name,
                        "op_type": op_type,
                        "op_category": kernel_def.get("op_category", ""),
                        "axes": kernel_def.get("axes", {}),
                        "test_configs": kernel_def.get("test_configs", []),
                        "baseline_ref": kernel_def.get("baseline_ref", {}),
                        "definition_file": str(json_file.relative_to(PROJECT_ROOT))
                    }
                    summary["operators"][op_type]["kernels"].append(kernel_info)
                    summary["operators"][op_type]["total_count"] += 1

                    # Add to model section
                    summary["models"][model_name]["operators"][op_type].append(kernel_info["name"])
                    summary["models"][model_name]["total_kernels"] += 1
                    summary["total_kernels"] += 1

    # Convert sets to lists for JSON serialization
    for op_type, op_data in summary["operators"].items():
        op_data["models"] = sorted(list(op_data["models"]))

    return summary

def format_model_name(model_name):
    """Format model name for display."""
    # Common model name mappings
    name_map = {
        "llama": "LLaMA",
        "llama3_8b": "LLaMA-3-8B",
        "qwen": "Qwen",
        "qwen3_4b": "Qwen3-4B",
        "deepseek": "DeepSeek",
        "mixtral": "Mixtral",
    }

    # Try exact match first
    if model_name in name_map:
        return name_map[model_name]

    # Try partial match
    for key, value in name_map.items():
        if key in model_name:
            return model_name.replace(key, value)

    # Default: capitalize and replace underscores
    return model_name.replace("_", "-").title()

def main():
    print("=" * 60)
    print("Generating kernels summary")
    print("=" * 60)

    summary = scan_definitions()

    # Print statistics
    print(f"\nTotal kernels: {summary['total_kernels']}")
    print(f"\nOperator types:")
    for op_type, op_data in summary["operators"].items():
        print(f"  {op_data['display_name']}: {op_data['total_count']} kernels")
        print(f"    Models: {', '.join(op_data['models'])}")

    print(f"\nModels:")
    for model, model_data in sorted(summary["models"].items()):
        print(f"  {model_data['display_name']}: {model_data['total_kernels']} kernels")
        print(f"    Operators: {', '.join(model_data['operators'].keys())}")

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"Output written to: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
