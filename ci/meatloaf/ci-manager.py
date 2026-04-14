#!/usr/bin/env python3
"""
meatloaf CI manager: compile, run, and track HIP vs SYCL benchmarks.

Modes:
  --seed    Build baseline from scratch (full suite)
  --check   Compare current results against baseline (PR-scoped or full)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def load_modules():
    """Load environment modules via shell"""
    env_setup = """
    set +u
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module purge 2>/dev/null || true
    module load llvm/22.0-native oneapi/2025.0.4 HIP/chipStar/2026.03.17 2>/dev/null || true
    export PATH=/space/pvelesko/install/oneapi/2025.0.4/bin:$PATH
    """
    # Just set PATH; modules loaded by shell parent
    os.environ['PATH'] = '/space/pvelesko/install/oneapi/2025.0.4/bin:' + os.environ.get('PATH', '')

def load_subset_json():
    """Load timing regexes from subset.json"""
    subset_path = Path('src/scripts/benchmarks/subset.json')
    if not subset_path.exists():
        return {}
    with open(subset_path) as f:
        return json.load(f)

def load_known_failures():
    """Load known_failures.yaml baseline"""
    yaml_path = Path('ci/meatloaf/known_failures.yaml')
    if not yaml_path.exists():
        return {'benchmarks': {}}

    # Simple YAML parser for our structure
    import yaml
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data or {'benchmarks': {}}
    except ImportError:
        print("WARNING: PyYAML not installed, using JSON fallback", file=sys.stderr)
        return {'benchmarks': {}}

def save_known_failures(data):
    """Save baseline to known_failures.yaml"""
    import yaml
    yaml_path = Path('ci/meatloaf/known_failures.yaml')
    data['meta']['generated_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def compile_benchmark(bench_dir, backend):
    """Compile a benchmark. Return 'compile' (fail) or None (success)"""
    cc = 'icpx' if backend == 'sycl' else 'hipcc'
    opts = 'CUDA=no HIP=no' if backend == 'sycl' else ''

    try:
        subprocess.run(['make', 'clean'], cwd=bench_dir, timeout=30,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        cmd = ['make', f'CC={cc}'] + (opts.split() if opts else [])
        result = subprocess.run(cmd, cwd=bench_dir, timeout=60,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None if result.returncode == 0 else 'compile'
    except subprocess.TimeoutExpired:
        return 'run'  # Treat compile timeout as run failure

def run_benchmark(bench_dir):
    """Run compiled benchmark. Return status and timing (if available)"""
    binary = bench_dir / 'main'
    if not binary.exists():
        return 'compile', None  # Shouldn't happen, but safety

    try:
        result = subprocess.run([str(binary)], cwd=bench_dir, timeout=60,
                               capture_output=True, text=True)
        status = 'pass' if result.returncode == 0 else 'correctness'
        return status, None
    except subprocess.TimeoutExpired:
        return 'run', None

def extract_timing(bench_dir, backend, subset_json):
    """Extract timing from benchmark output via subset.json regex"""
    bench_name = bench_dir.name.replace(f'-{backend}', '')
    if bench_name not in subset_json:
        return None

    regex_pattern = subset_json[bench_name][0]
    if not regex_pattern:
        return None

    try:
        binary = bench_dir / 'main'
        result = subprocess.run([str(binary)], cwd=bench_dir, timeout=60,
                               capture_output=True, text=True)
        matches = re.findall(regex_pattern, result.stdout)
        if matches:
            times = [float(m) for m in matches if m]
            return sum(times) / len(times) if times else None
    except Exception:
        pass
    return None

def test_benchmark_pair(base_name, repo_root, subset_json):
    """Test both HIP and SYCL variants. Return {backend: (status, time)}"""
    results = {}

    for backend in ['hip', 'sycl']:
        bench_dir = repo_root / f'{base_name}-{backend}'
        if not bench_dir.exists():
            results[backend] = ('absent', None)
            continue

        # Check known failures for skip
        baseline = load_known_failures()
        bench_baseline = baseline.get('benchmarks', {}).get(base_name, {})
        if backend in bench_baseline and bench_baseline[backend] == 'skip':
            results[backend] = ('skip', None)
            continue

        # Compile
        compile_status = compile_benchmark(bench_dir, backend)
        if compile_status:
            results[backend] = (compile_status, None)
            continue

        # Run
        run_status, _ = run_benchmark(bench_dir)
        timing = None
        if run_status == 'pass':
            timing = extract_timing(bench_dir, backend, subset_json)

        results[backend] = (run_status, timing)

    return results

def compare_status(baseline_status, current_status):
    """Return 'regression', 'improvement', 'unchanged'"""
    severity = {'skip': 0, 'pass': 1, 'perf_discrepancy': 2,
                'correctness': 3, 'run': 4, 'compile': 5, 'absent': 6}
    base_sev = severity.get(baseline_status, 6)
    curr_sev = severity.get(current_status, 6)

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

    subset_json = load_subset_json()
    baseline = load_known_failures()

    # Detect benchmarks to test
    bench_list_path = os.getenv('BENCH_LIST')
    if bench_list_path and Path(bench_list_path).exists():
        with open(bench_list_path) as f:
            benches = [line.strip() for line in f if line.strip()]
    else:
        # All benchmarks
        benches = sorted(set(
            d.name.replace('-hip', '').replace('-sycl', '')
            for d in Path('.').glob('*-hip') | set(Path('.').glob('*-sycl'))
        ))

    print(f"Testing {len(benches)} benchmarks...")

    results = {'benchmarks': {}}
    regressions = []
    improvements = []

    for bench in benches:
        pair_results = test_benchmark_pair(bench, repo_root, subset_json)
        results['benchmarks'][bench] = {
            backend: {'status': status, 'time': timing}
            for backend, (status, timing) in pair_results.items()
        }

        # Check for regressions
        for backend, (status, timing) in pair_results.items():
            baseline_entry = baseline.get('benchmarks', {}).get(bench, {}).get(backend)
            if baseline_entry:
                verdict = compare_status(baseline_entry, status)
                if verdict == 'regression':
                    regressions.append((bench, backend, baseline_entry, status))
                elif verdict == 'improvement':
                    improvements.append((bench, backend, baseline_entry, status))

    # Generate report
    report_path = Path('ci/meatloaf/results')
    report_path.mkdir(parents=True, exist_ok=True)

    with open(report_path / 'report.md', 'w') as f:
        f.write("# HIP vs SYCL Build/Run Test\n\n")
        if regressions:
            f.write(f"## 🚨 Regressions ({len(regressions)})\n")
            for bench, backend, baseline_status, current_status in regressions:
                f.write(f"- {bench}/{backend}: {baseline_status} → {current_status}\n")
            f.write("\n")
        if improvements:
            f.write(f"## 📈 Improvements ({len(improvements)})\n")
            for bench, backend, baseline_status, current_status in improvements:
                f.write(f"- {bench}/{backend}: {baseline_status} → {current_status}\n")
            f.write("\n")
        f.write(f"Tested {len(benches)} benchmarks. ")
        f.write(f"Regressions: {len(regressions)}, Improvements: {len(improvements)}\n")

    if args.seed:
        results['meta'] = baseline.get('meta', {})
        save_known_failures(results)
        print(f"Baseline updated: {len(benches)} benchmarks")
        return 0
    else:
        print(f"Report: {len(regressions)} regressions, {len(improvements)} improvements")
        if regressions:
            print("Exiting with error due to regressions")
            return 1
        return 0

if __name__ == '__main__':
    sys.exit(main())
