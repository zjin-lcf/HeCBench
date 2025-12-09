#!/usr/bin/env python3
"""
generate_metadata.py - Generate benchmarks.yaml from existing metadata

This script reads the existing subset.json file and benchmark CMakeLists.txt
files to generate a comprehensive benchmarks.yaml metadata file.

Usage:
    python3 generate_metadata.py [--output benchmarks.yaml]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Constants
SCRIPT_DIR = Path(__file__).parent
HECBENCH_ROOT = SCRIPT_DIR.parent
SRC_DIR = HECBENCH_ROOT / "src"
SUBSET_JSON = SRC_DIR / "scripts" / "benchmarks" / "subset.json"

MODELS = ["cuda", "hip", "sycl", "omp"]

# Category mappings based on common benchmark patterns
CATEGORY_PATTERNS = {
    # Machine Learning
    r'(adam|adamw|attention|backprop|bn|crossEntropy|dense-embedding|dropout|'
    r'gelu|glu|groupnorm|knn|layernorm|logprob|lr|mnist|multinomial|nlll|'
    r'perplexity|rmsnorm|softmax|swish|transformer)': ['machinelearning'],

    # Imaging/Graphics
    r'(bilateral|boxfilter|dct|dxtc|entropy|gabor|gamma|gaussian|histogram|'
    r'mandelbrot|medianfilter|overlay|resize|srad|tonemapping)': ['imaging'],

    # Simulation
    r'(bh|cobahh|fdtd|heat|hotspot|ising|jacobi|laplace|lavaMD|lbm|md|'
    r'minimod|nbody|particlefilter|pathfinder|s3d|sheath|stencil)': ['simulation'],

    # Graph algorithms
    r'(bfs|floydwarshall|page-rank|sssp)': ['graph'],

    # Cryptography
    r'(aes|ecdh|md5|merkle|murmurhash|sha)': ['cryptography'],

    # Math/Linear Algebra
    r'(blas|cholesky|crs|eigenvalue|fft|fwt|gemm|gemv|lu|qr|svd|tridiagonal)': ['math'],

    # Compression
    r'(ans|bwt|huffman|lzss|rle)': ['compression'],

    # Bioinformatics
    r'(dna|epistasis|frna|hmm|nw|sw)': ['bioinformatics'],
}


def load_subset_json() -> dict:
    """Load the existing subset.json benchmark metadata."""
    if not SUBSET_JSON.exists():
        print(f"Warning: {SUBSET_JSON} not found")
        return {}

    with open(SUBSET_JSON) as f:
        return json.load(f)


def discover_benchmarks() -> dict:
    """Discover all benchmarks and their implementations."""
    benchmarks = defaultdict(lambda: {"models": [], "categories": set()})

    for entry in SRC_DIR.iterdir():
        if not entry.is_dir():
            continue

        name = entry.name
        model = None
        for m in MODELS:
            suffix = f"-{m}"
            if name.endswith(suffix):
                model = m
                name = name[:-len(suffix)]
                break

        if model is None:
            continue

        benchmarks[name]["models"].append(model)

        # Try to extract categories from CMakeLists.txt
        cmake_file = entry / "CMakeLists.txt"
        if cmake_file.exists():
            content = cmake_file.read_text()
            cat_match = re.search(r'CATEGORIES\s+([^\)]+)', content)
            if cat_match:
                cats = cat_match.group(1).split()
                benchmarks[name]["categories"].update(cats)

    return benchmarks


def infer_categories(name: str) -> list:
    """Infer categories based on benchmark name patterns."""
    categories = []
    name_lower = name.lower()

    for pattern, cats in CATEGORY_PATTERNS.items():
        if re.search(pattern, name_lower):
            categories.extend(cats)

    # Default to 'algorithms' if no category found
    if not categories:
        categories = ['algorithms']

    return list(set(categories))


def generate_yaml(benchmarks: dict, metadata: dict, output_path: Path):
    """Generate the benchmarks.yaml file."""

    lines = [
        "# HeCBench Benchmark Metadata",
        "#",
        "# Auto-generated from subset.json and CMakeLists.txt files",
        "# Manual edits may be required for accuracy",
        "#",
        "# Format:",
        "#   benchmark_name:",
        "#     categories: [list of categories]",
        "#     description: Brief description",
        "#     models: [available implementations]",
        "#     test:",
        "#       regex: Regular expression to match success output",
        "#       args: [list of command-line arguments for testing]",
        "#       timeout: Timeout in seconds (default: 300)",
        "#",
        "",
    ]

    # Sort benchmarks alphabetically
    for name in sorted(benchmarks.keys()):
        bench = benchmarks[name]

        # Get categories from CMake or infer
        categories = list(bench["categories"]) if bench["categories"] else infer_categories(name)

        # Get test metadata from subset.json
        test_info = metadata.get(name)

        lines.append(f"{name}:")
        lines.append(f"  categories: [{', '.join(sorted(categories))}]")
        lines.append(f"  models: [{', '.join(sorted(bench['models']))}]")

        if test_info:
            regex = test_info[0] if len(test_info) > 0 else ""
            args = test_info[1] if len(test_info) > 1 else []
            binary = test_info[2] if len(test_info) > 2 else "main"

            lines.append("  test:")
            # Escape single quotes in regex for YAML
            regex_escaped = regex.replace("'", "''")
            lines.append(f"    regex: '{regex_escaped}'")

            if args:
                # Format args as YAML list
                args_str = ", ".join(f'"{a}"' for a in args)
                lines.append(f"    args: [{args_str}]")
            else:
                lines.append("    args: []")

            if binary != "main":
                lines.append(f"    binary: {binary}")

            lines.append("    timeout: 300")
        else:
            lines.append("  # No test metadata available")

        lines.append("")

    # Write output
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated {output_path}")
    print(f"  Total benchmarks: {len(benchmarks)}")
    print(f"  With test metadata: {sum(1 for n in benchmarks if n in metadata)}")
    print(f"  Without test metadata: {sum(1 for n in benchmarks if n not in metadata)}")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmarks.yaml metadata')
    parser.add_argument('-o', '--output', default=str(HECBENCH_ROOT / 'benchmarks.yaml'),
                        help='Output file path')
    args = parser.parse_args()

    print("Loading existing metadata...")
    metadata = load_subset_json()
    print(f"  Found {len(metadata)} entries in subset.json")

    print("Discovering benchmarks...")
    benchmarks = discover_benchmarks()
    print(f"  Found {len(benchmarks)} unique benchmarks")

    print("Generating YAML...")
    generate_yaml(benchmarks, metadata, Path(args.output))

    return 0


if __name__ == '__main__':
    sys.exit(main())
