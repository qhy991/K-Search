#!/usr/bin/env python3
"""
Generate spec.json from definition files and implementation overrides.

This tool reduces duplication by generating spec.json files from:
1. Definition files in definitions/quant_gemm/
2. Implementation-specific overrides in impl.json

Usage:
    # Generate all spec.json files
    python tools/generate_spec.py --all

    # Generate from a specific definition
    python tools/generate_spec.py --definition definitions/quant_gemm/deepseek_v3/w8a8c8_q8_0_fp32_int8_ds3_att_out.json

    # Validate existing spec.json against definitions
    python tools/generate_spec.py --validate

    # Show what would be generated (dry run)
    python tools/generate_spec.py --all --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import fnmatch


@dataclass
class ValidationResult:
    """Result of a validation check."""
    path: str
    valid: bool
    message: str
    details: Optional[Dict] = None


class SpecGenerator:
    """
    Generator for spec.json files from definitions and impl overrides.

    Reads definition files from definitions/quant_gemm/ and merges them with
    implementation-specific fields from impl.json files in variant directories.
    """

    def __init__(self, project_root: Path):
        """
        Initialize the spec generator.

        Args:
            project_root: Root directory of the KernelEvalPlus project
        """
        self.project_root = project_root
        self.definitions_dir = project_root / 'definitions'
        self.variants_dir = project_root / 'python' / 'operators' / 'quant_gemm' / 'variants'

    def find_definitions(self, pattern: Optional[str] = None) -> List[Path]:
        """
        Find all definition files.

        Args:
            pattern: Optional glob pattern to filter definitions

        Returns:
            List of paths to definition files
        """
        definitions = []

        for def_file in self.definitions_dir.rglob('*.json'):
            # Skip template files by checking path
            if 'templates' in str(def_file):
                continue

            if pattern:
                if not fnmatch.fnmatch(def_file.name, pattern):
                    continue

            definitions.append(def_file)

        return sorted(definitions)

    def find_variant_by_name(self, name: str) -> Optional[Path]:
        """
        Find a variant directory by name.

        Args:
            name: Variant name (e.g., 'w8a8c8_q8_0_fp32_int8')

        Returns:
            Path to variant directory or None if not found
        """
        # Check direct match
        for variant_dir in self.variants_dir.rglob('spec.json'):
            spec_dir = variant_dir.parent
            try:
                with open(variant_dir) as f:
                    spec = json.load(f)
                    if spec.get('name') == name:
                        return spec_dir
            except (json.JSONDecodeError, IOError):
                continue

            # Also check directory name
            if spec_dir.name == name:
                return spec_dir

        return None

    def load_definition(self, def_path: Path) -> Dict:
        """
        Load a definition file.

        Args:
            def_path: Path to definition JSON file

        Returns:
            Dictionary with definition contents
        """
        with open(def_path) as f:
            return json.load(f)

    def load_impl(self, variant_dir: Path) -> Optional[Dict]:
        """
        Load implementation overrides from impl.json.

        Args:
            variant_dir: Path to variant directory

        Returns:
            Dictionary with impl contents or None if not found
        """
        impl_path = variant_dir / 'impl.json'
        if not impl_path.exists():
            return None

        with open(impl_path) as f:
            return json.load(f)

    def definition_to_spec(self, definition: Dict, impl: Optional[Dict] = None) -> Dict:
        """
        Convert a definition to spec.json format.

        Args:
            definition: Definition dictionary
            impl: Optional implementation overrides

        Returns:
            Spec dictionary ready for JSON serialization
        """
        spec = {}

        # Identity fields
        spec['name'] = definition.get('name', '')
        spec['family'] = definition.get('op_category', definition.get('op_type', 'quant_gemm'))
        spec['version'] = definition.get('version', '1.0.0')
        spec['description'] = definition.get('description', '')

        # Kernel info (from impl if available)
        if impl and 'kernel' in impl:
            spec['kernel'] = impl['kernel']
        elif 'kernel' in definition:
            spec['kernel'] = definition['kernel']
        else:
            # Generate default kernel info based on name
            name = spec['name']
            spec['kernel'] = {
                'file': 'kernel.cu',
                'entry_point': f'gemm_{name}'
            }

        # Input/output specifications
        spec['inputs'] = self._convert_io(definition.get('inputs', {}), definition.get('types', {}))
        spec['outputs'] = self._convert_io(definition.get('outputs', {}), definition.get('types', {}))

        # Parameters
        spec['params'] = self._convert_params(definition.get('axes', {}))

        # Reference implementation
        if impl and 'reference' in impl:
            spec['reference'] = impl['reference']
        else:
            spec['reference'] = 'reference.py:run'

        # Test configs
        spec['test_configs'] = definition.get('test_configs', [
            {'name': 'single', 'M': 1, 'N': 4096, 'K': 4096},
            {'name': 'small_batch', 'M': 4, 'N': 4096, 'K': 4096},
        ])

        # Accuracy spec
        spec['accuracy'] = {
            'metric': 'nmse',
            'threshold': 0.1
        }

        # Formula
        if 'formula' in definition:
            spec['formula'] = definition['formula']

        # Add any additional test configs from impl
        if impl and 'additional_test_configs' in impl:
            spec['test_configs'].extend(impl['additional_test_configs'])

        return spec

    def _convert_io(self, io_spec: Dict, types: Dict) -> Dict:
        """
        Convert definition IO spec to spec.json format.

        Args:
            io_spec: Input/output specification from definition
            types: Type definitions from definition

        Returns:
            Converted IO specification
        """
        result = {}

        for name, info in io_spec.items():
            entry = {
                'dtype': info.get('dtype', 'float32'),
                'shape': info.get('shape', []),
                'description': info.get('description', ''),
            }

            # Add quantizer info if it's a quantized type
            dtype = entry['dtype']
            if dtype.startswith('block_q'):
                # Map dtype to quantizer name
                quant_name = dtype.replace('block_', '').replace('_', '')
                entry['quantizer'] = f'quantize_{quant_name}'

            result[name] = entry

        return result

    def _convert_params(self, axes: Dict) -> Dict:
        """
        Convert definition axes to spec.json params format.

        Args:
            axes: Axes specification from definition

        Returns:
            Converted params specification
        """
        result = {}

        for name, info in axes.items():
            if name in ('block_size',):  # Skip internal axes
                continue

            entry = {
                'type': 'int',
                'description': info.get('description', ''),
            }

            # Add default value for const axes
            if info.get('type') == 'const':
                entry['default'] = info.get('value')

            # Add constraints
            if name == 'K':
                entry['constraint'] = 'K % 32 == 0'
            elif name in ('M', 'N'):
                entry['constraint'] = f'{name} >= 1'

            result[name] = entry

        return result

    def generate_spec(self, def_path: Path, variant_dir: Optional[Path] = None,
                      dry_run: bool = False) -> Tuple[Path, Dict]:
        """
        Generate spec.json from a definition file.

        Args:
            def_path: Path to definition file
            variant_dir: Path to variant directory (auto-detected if None)
            dry_run: If True, don't write file

        Returns:
            Tuple of (output_path, spec_dict)
        """
        definition = self.load_definition(def_path)
        name = definition.get('name', def_path.stem)

        # Find or create variant directory
        if variant_dir is None:
            # Try to find existing variant
            base_name = name.split('_ds3')[0]  # Remove model-specific suffix
            variant_dir = self.find_variant_by_name(base_name)

            if variant_dir is None:
                # Create path based on definition location
                rel_path = def_path.relative_to(self.definitions_dir / 'quant_gemm')
                variant_dir = self.variants_dir / rel_path.parent / base_name

        # Load impl.json if exists
        impl = self.load_impl(variant_dir)

        # Generate spec
        spec = self.definition_to_spec(definition, impl)

        # Output path
        output_path = variant_dir / 'spec.json'

        if not dry_run:
            # Create directory if needed
            variant_dir.mkdir(parents=True, exist_ok=True)

            # Write spec.json
            with open(output_path, 'w') as f:
                json.dump(spec, f, indent=2)

        return output_path, spec

    def generate_all(self, pattern: Optional[str] = None,
                     dry_run: bool = False) -> List[Tuple[Path, Dict]]:
        """
        Generate all spec.json files from definitions.

        Args:
            pattern: Optional glob pattern to filter definitions
            dry_run: If True, don't write files

        Returns:
            List of (output_path, spec_dict) tuples
        """
        results = []

        for def_path in self.find_definitions(pattern):
            try:
                result = self.generate_spec(def_path, dry_run=dry_run)
                results.append(result)
            except Exception as e:
                print(f"Error processing {def_path}: {e}")

        return results

    def validate_spec(self, spec_path: Path) -> ValidationResult:
        """
        Validate a spec.json file.

        Args:
            spec_path: Path to spec.json file

        Returns:
            ValidationResult with validation status
        """
        try:
            with open(spec_path) as f:
                spec = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return ValidationResult(
                path=str(spec_path),
                valid=False,
                message=f"Failed to load: {e}"
            )

        # Check required fields
        required = ['name', 'family', 'inputs', 'outputs', 'params']
        missing = [f for f in required if f not in spec]

        if missing:
            return ValidationResult(
                path=str(spec_path),
                valid=False,
                message=f"Missing required fields: {missing}"
            )

        # Check kernel info
        if 'kernel' not in spec:
            return ValidationResult(
                path=str(spec_path),
                valid=False,
                message="Missing kernel specification"
            )

        # Check test configs
        if not spec.get('test_configs'):
            return ValidationResult(
                path=str(spec_path),
                valid=False,
                message="No test configurations defined"
            )

        # All checks passed
        return ValidationResult(
            path=str(spec_path),
            valid=True,
            message="Valid"
        )

    def validate_all(self) -> List[ValidationResult]:
        """
        Validate all spec.json files.

        Returns:
            List of ValidationResult for each spec file
        """
        results = []

        for spec_path in self.variants_dir.rglob('spec.json'):
            result = self.validate_spec(spec_path)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate spec.json from definition files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python tools/generate_spec.py --all                  # Generate all
  python tools/generate_spec.py --definition def.json  # Generate one
  python tools/generate_spec.py --validate             # Validate all
  python tools/generate_spec.py --all --dry-run        # Preview changes
'''
    )

    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate all spec.json files from definitions')
    parser.add_argument('--definition', '-d', type=str,
                        help='Generate from specific definition file')
    parser.add_argument('--pattern', '-p', type=str,
                        help='Filter definitions by pattern')
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Validate existing spec.json files')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be generated without writing')
    parser.add_argument('--project-root', type=str,
                        help='Project root directory (auto-detected if not specified)')

    args = parser.parse_args()

    # Find project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Auto-detect from script location
        project_root = Path(__file__).parent.parent.parent

    generator = SpecGenerator(project_root)

    # Handle --validate
    if args.validate:
        print("Validating spec.json files...")
        print()

        results = generator.validate_all()

        valid_count = sum(1 for r in results if r.valid)
        invalid_count = len(results) - valid_count

        for result in results:
            status = "[OK]" if result.valid else "[FAIL]"
            rel_path = Path(result.path).relative_to(project_root)
            print(f"{status} {rel_path}: {result.message}")

        print()
        print(f"Results: {valid_count} valid, {invalid_count} invalid")

        return 0 if invalid_count == 0 else 1

    # Handle --definition
    if args.definition:
        def_path = Path(args.definition)
        if not def_path.is_absolute():
            def_path = project_root / def_path

        if not def_path.exists():
            print(f"Error: Definition file not found: {def_path}")
            return 1

        print(f"Generating spec from: {def_path}")

        output_path, spec = generator.generate_spec(def_path, dry_run=args.dry_run)

        if args.dry_run:
            print(f"Would write to: {output_path}")
            print(json.dumps(spec, indent=2))
        else:
            print(f"Generated: {output_path}")

        return 0

    # Handle --all
    if args.all:
        print("Generating all spec.json files...")
        if args.dry_run:
            print("(dry run - no files will be written)")
        print()

        results = generator.generate_all(pattern=args.pattern, dry_run=args.dry_run)

        for output_path, spec in results:
            rel_path = output_path.relative_to(project_root)
            action = "Would generate" if args.dry_run else "Generated"
            print(f"{action}: {rel_path}")

        print()
        print(f"Total: {len(results)} spec files")

        return 0

    # No action specified
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
