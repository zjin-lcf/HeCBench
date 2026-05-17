#!/usr/bin/env python3
"""
meatloaf CI manager: compile and run HIP vs SYCL benchmarks.

Tracks pass/fail status (compile, run, correctness) and detects regressions.

Modes:
  --seed    Build baseline from scratch (full suite)
  --check   Check for regressions against baseline (PR-scoped or full)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def load_modules():
    """Set up environment (modules loaded by caller)"""
    os.environ['PATH'] = '/space/pvelesko/install/oneapi/2025.0.4/bin:' + os.environ.get('PATH', '')

def load_known_failures():
    """Load known_failures.yaml baseline"""
    yaml_path = Path('ci/meatloaf/known_failures.yaml')
    if not yaml_path.exists():
        return {'benchmarks': {}}

    # Simple YAML parser for our structure
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data or {'benchmarks': {}}
    except ImportError:
        print("WARNING: PyYAML not installed, using empty baseline", file=sys.stderr)
        return {'benchmarks': {}}

def save_known_failures(data):
    """Save baseline to known_failures.yaml"""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required to save baseline", file=sys.stderr)
        sys.exit(1)

    yaml_path = Path('ci/meatloaf/known_failures.yaml')
    data['meta']['generated_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def compile_benchmark(bench_dir, backend):
    """
    Compile a benchmark.
    Return: 'compile' (fail), or None (success)
    """
    cc = 'icpx' if backend == 'sycl' else 'hipcc'
    opts = 'CUDA=no HIP=no' if backend == 'sycl' else ''

    try:
        # Clean
        subprocess.run(['make', 'clean'], cwd=bench_dir, timeout=30,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        # Compile
        cmd = ['make', f'CC={cc}'] + (opts.split() if opts else [])
        result = subprocess.run(cmd, cwd=bench_dir, timeout=60,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None if result.returncode == 0 else 'compile'
    except subprocess.TimeoutExpired:
        return 'run'  # Compile timeout → run failure

def run_benchmark(bench_dir):
    """
    Run compiled benchmark.
    Return: 'pass', 'run' (timeout), or 'correctness' (non-zero exit)
    """
    binary = bench_dir / 'main'
    if not binary.exists():
        return 'compile'  # Shouldn't happen

    try:
        result = subprocess.run([str(binary)], cwd=bench_dir, timeout=60,
                               capture_output=True, text=True)
        return 'pass' if result.returncode == 0 else 'correctness'
    except subprocess.TimeoutExpired:
        return 'run'

def test_benchmark_pair(base_name, repo_root):
    """
    Test both HIP and SYCL variants.
    Return: {backend: status}
    """
    results = {}

    for backend in ['hip', 'sycl']:
        bench_dir = repo_root / f'{base_name}-{backend}'
        if not bench_dir.exists():
            results[backend] = 'absent'
            continue

        # Check known failures for skip
        baseline = load_known_failures()
        bench_baseline = baseline.get('benchmarks', {}).get(base_name, {})
        if isinstance(bench_baseline, dict) and bench_baseline.get(backend) == 'skip':
            results[backend] = 'skip'
            continue

        # Compile
        compile_status = compile_benchmark(bench_dir, backend)
        if compile_status:
            results[backend] = compile_status
            continue

        # Run
        run_status = run_benchmark(bench_dir)
        results[backend] = run_status

    return results

def compare_status(baseline_status, current_status):
    """
    Regression severity (worst to best):
      skip > pass > correctness > run > compile > absent

    Return: 'regression', 'improvement', or 'unchanged'
    """
    severity = {
        'skip': 0,
        'pass': 1,
        'correctness': 2,
        'run': 3,
        'compile': 4,
        'absent': 5
    }

    base_sev = severity.get(baseline_status, 5)
    curr_sev = severity.get(current_status, 5)

    if curr_sev < base_sev:
        return 'regression'
    elif curr_sev > base_sev:
        return 'improvement'
    return 'unchanged'

def main():
    parser = argparse.ArgumentParser(description='meatloaf CI manager')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seed', action='store_true',
                      help='Regenerate known_failures.yaml baseline')
    group.add_argument('--check', action='store_true',
                      help='Check for regressions against baseline')
    args = parser.parse_args()

    load_modules()
    repo_root = Path(os.getenv('CI_ROOT', '.')).resolve()
    os.chdir(repo_root)

    baseline = load_known_failures()

    # Detect benchmarks to test
    bench_list_path = os.getenv('BENCH_LIST')
    if bench_list_path and Path(bench_list_path).exists():
        with open(bench_list_path) as f:
            benches = [line.strip() for line in f if line.strip()]
    else:
        # All benchmarks: union of *-hip and *-sycl
        hip_benches = set(d.name[:-4] for d in Path('.').glob('*-hip'))
        sycl_benches = set(d.name[:-5] for d in Path('.').glob('*-sycl'))
        benches = sorted(hip_benches | sycl_benches)

    print(f"Testing {len(benches)} benchmarks...")

    results = {'benchmarks': {}}
    regressions = []
    improvements = []

    for bench in benches:
        pair_results = test_benchmark_pair(bench, repo_root)
        results['benchmarks'][bench] = pair_results

        # Check for regressions
        for backend, status in pair_results.items():
            baseline_entry = baseline.get('benchmarks', {}).get(bench, {})
            baseline_status = baseline_entry.get(backend) if isinstance(baseline_entry, dict) else baseline_entry

            if baseline_status:
                verdict = compare_status(baseline_status, status)
                if verdict == 'regression':
                    regressions.append((bench, backend, baseline_status, status))
                elif verdict == 'improvement':
                    improvements.append((bench, backend, baseline_status, status))

    # Generate report
    report_path = Path('ci/meatloaf/results')
    report_path.mkdir(parents=True, exist_ok=True)

    with open(report_path / 'report.md', 'w') as f:
        f.write("# HIP vs SYCL Benchmark Test\n\n")

        if regressions:
            f.write(f"## 🚨 Regressions ({len(regressions)})\n\n")
            for bench, backend, baseline_status, current_status in regressions:
                f.write(f"- **{bench}/{backend}**: {baseline_status} → {current_status}\n")
            f.write("\n")
        else:
            f.write("## ✓ No regressions\n\n")

        if improvements:
            f.write(f"## 📈 Improvements ({len(improvements)})\n\n")
            for bench, backend, baseline_status, current_status in improvements:
                f.write(f"- **{bench}/{backend}**: {baseline_status} → {current_status}\n")
            f.write("\n")

        f.write(f"**Summary**: Tested {len(benches)} benchmarks. ")
        f.write(f"Regressions: {len(regressions)}, Improvements: {len(improvements)}\n")

    # Save results JSON
    with open(report_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if args.seed:
        results['meta'] = baseline.get('meta', {})
        save_known_failures(results)
        print(f"✓ Baseline updated: {len(benches)} benchmarks")
        return 0
    else:
        print(f"Report: {len(regressions)} regressions, {len(improvements)} improvements")
        if regressions:
            print("✗ Exiting with error due to regressions")
            return 1
        print("✓ No regressions")
        return 0

if __name__ == '__main__':
    sys.exit(main())
