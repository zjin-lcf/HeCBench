#!/usr/bin/env python3
"""
Script to generate CMakeLists.txt files for HeCBench benchmarks
"""

import os
import sys
from pathlib import Path

def get_source_files(benchmark_dir):
    """Get list of source files (.cu or .cpp) in benchmark directory"""
    sources = []
    for ext in ['*.cu', '*.cpp']:
        sources.extend(Path(benchmark_dir).glob(ext))
    # Return just the filenames, not full paths
    return [s.name for s in sorted(sources)]

def get_categories(benchmark_name):
    """Infer categories from benchmark name - this is a simplified heuristic"""
    # Common patterns
    ml_keywords = ['adam', 'adamw', 'attention', 'softmax', 'backprop', 'lstm', 'gru']
    crypto_keywords = ['aes', 'md5', 'sha', 'rsa']
    math_keywords = ['fft', 'dct', 'gemm', 'blas']
    graph_keywords = ['bfs', 'dfs', 'floyd', 'pagerank']

    categories = []

    if any(kw in benchmark_name.lower() for kw in ml_keywords):
        categories.append('ml')
    if any(kw in benchmark_name.lower() for kw in crypto_keywords):
        categories.append('cryptography')
    if any(kw in benchmark_name.lower() for kw in math_keywords):
        categories.append('math')
    if any(kw in benchmark_name.lower() for kw in graph_keywords):
        categories.append('graph')

    # Default category if none matched
    if not categories:
        categories.append('algorithms')

    return categories

def create_cmake_file(benchmark_name, model, source_files, categories):
    """Generate CMakeLists.txt content"""
    sources_str = ' '.join(source_files)
    categories_str = ' '.join(categories)

    content = f"""# {benchmark_name}-{model}/CMakeLists.txt

add_hecbench_benchmark(
    NAME {benchmark_name}
    MODEL {model}
    SOURCES {sources_str}
    CATEGORIES {categories_str}
)
"""
    return content

def main():
    src_dir = Path('src')

    if not src_dir.exists():
        # Try current directory
        if Path('.').resolve().name == 'src':
            src_dir = Path('.')
        else:
            print("Error: Must run from HeCBench root or src directory")
            sys.exit(1)

    benchmarks_to_convert = [
        # Batch 11: O-P benchmarks (27 benchmarks)
        'openmp', 'opticalFlow', 'overlap', 'overlay', 'p2p', 'p4', 'pad',
        'page-rank', 'particle-diffusion', 'particles', 'pcc', 'perlin',
        'permutate', 'permute', 'phmm', 'pingpong', 'pitch', 'pnpoly', 'pns',
        'pointwise', 'pool', 'popcount', 'prefetch', 'present', 'prna',
        'projectile', 'pso'
    ]

    models = ['cuda', 'hip', 'sycl', 'omp']

    created_count = 0

    for benchmark in benchmarks_to_convert:
        categories = get_categories(benchmark)

        for model in models:
            bench_dir = src_dir / f'{benchmark}-{model}'

            if not bench_dir.exists():
                print(f"  Skipping {benchmark}-{model} (directory not found)")
                continue

            # Check if CMakeLists.txt already exists
            cmake_file = bench_dir / 'CMakeLists.txt'
            if cmake_file.exists():
                print(f"  Skipping {benchmark}-{model} (CMakeLists.txt exists)")
                continue

            # Get source files
            sources = get_source_files(bench_dir)
            if not sources:
                print(f"  WARNING: {benchmark}-{model} has no source files!")
                continue

            # Create CMakeLists.txt
            content = create_cmake_file(benchmark, model, sources, categories)
            cmake_file.write_text(content)
            created_count += 1
            print(f"✓ Created {benchmark}-{model}/CMakeLists.txt ({len(sources)} sources)")

    print(f"\nCreated {created_count} CMakeLists.txt files")

    # Update src/CMakeLists.txt with new benchmarks
    src_cmake = src_dir / 'CMakeLists.txt'
    if src_cmake.exists():
        content = src_cmake.read_text()

        # Find the HECBENCH_POC_BENCHMARKS list
        if 'HECBENCH_POC_BENCHMARKS' in content:
            # Add new benchmarks
            import re
            benchmarks_list = '\n    '.join(benchmarks_to_convert)

            # This is a manual step - tell user to update
            print(f"\n⚠ Don't forget to update src/CMakeLists.txt to include:")
            print(f"   {', '.join(benchmarks_to_convert)}")

if __name__ == '__main__':
    main()
