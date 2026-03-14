#!/usr/bin/env python3
"""
Build this variant (delegates to central builder).

Usage:
    python build.py           # Build this variant
    python build.py --verbose # With verbose output

For building multiple variants, use: python build_variants.py
"""

import argparse
import sys
from pathlib import Path

VARIANT_DIR = Path(__file__).parent.resolve()
# 6 levels up: variants/deepseek_v3/w8a8c8_q8_0_q8_1 -> ... -> project root
PROJECT_ROOT = VARIANT_DIR.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from build_variants import VariantBuilder
from build_config import BuildConfig


def main():
    parser = argparse.ArgumentParser(description='Build this variant')
    parser.add_argument('--jit', action='store_true', default=True)
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    builder = VariantBuilder(BuildConfig())
    method = 'setup' if args.setup else 'jit'
    return 0 if builder.build_variant(VARIANT_DIR, method, args.verbose) else 1


if __name__ == '__main__':
    sys.exit(main())
