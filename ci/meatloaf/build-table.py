#!/usr/bin/env python3
"""
Build a side-by-side markdown comparison table from two autohecbench.py runs:
chipStar HIP and SYCL (icpx). Designed to be appended to $GITHUB_STEP_SUMMARY
by the meatloaf CI job.

Inputs (per backend):
- CSV (--hip-csv / --sycl-csv): one line per benchmark,
  '<bench-name>, <run1>, <run2>, ...' as written by autohecbench.py.
  Times are milliseconds (the regex in subset.json captures ms).
- Summary JSON (--hip-summary / --sycl-summary):
  {'<bench-name>': {'compile': 'success|failed', 'run': 'success|failed|skipped'}}.

Output: a markdown file the workflow appends to GITHUB_STEP_SUMMARY.
Always exits 0; the table itself is the report.
"""

import argparse
import csv
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_csv(path: Path) -> dict:
    """Return {bench_name: mean_ms} for benchmarks that produced timings."""
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            times = []
            for cell in row[1:]:
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    times.append(float(cell))
                except ValueError:
                    pass
            if times:
                out[name] = sum(times) / len(times)
    return out


def parse_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def status_glyph(summary: dict, key: str) -> str:
    entry = summary.get(key)
    if not entry:
        return "—"
    compile_ = entry.get("compile", "?")
    run = entry.get("run", "?")
    if compile_ != "success":
        return "✗ build"
    if run == "success":
        return "✓"
    if run == "skipped":
        return "⏭"
    return "✗ run"


def fmt_time(value):
    if value is None:
        return "—"
    if value < 1.0:
        return f"{value:.4f}"
    if value < 100:
        return f"{value:.3f}"
    return f"{value:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip-csv", required=True, type=Path)
    ap.add_argument("--hip-summary", required=True, type=Path)
    ap.add_argument("--sycl-csv", required=True, type=Path)
    ap.add_argument("--sycl-summary", required=True, type=Path)
    ap.add_argument("--subset", required=True, type=Path,
                    help="Curated subset list (one base benchmark name per line)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    hip_times = parse_csv(args.hip_csv)
    sycl_times = parse_csv(args.sycl_csv)
    hip_sum = parse_summary(args.hip_summary)
    sycl_sum = parse_summary(args.sycl_summary)

    subset = []
    for line in args.subset.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            subset.append(line)

    rows = []
    hip_pass = sycl_pass = 0
    for base in subset:
        hip_key = f"{base}-hip"
        sycl_key = f"{base}-sycl"
        hip_t = hip_times.get(hip_key)
        sycl_t = sycl_times.get(sycl_key)
        hip_glyph = status_glyph(hip_sum, hip_key)
        sycl_glyph = status_glyph(sycl_sum, sycl_key)
        if hip_glyph == "✓":
            hip_pass += 1
        if sycl_glyph == "✓":
            sycl_pass += 1
        if hip_t and sycl_t and hip_t > 0:
            ratio = sycl_t / hip_t
            speedup = f"{ratio:.2f}x"
        else:
            speedup = "—"
        rows.append((base, hip_glyph, fmt_time(hip_t), sycl_glyph, fmt_time(sycl_t), speedup))

    lines = []
    lines.append("# chipStar HIP vs SYCL — meatloaf CI")
    lines.append("")
    lines.append(f"- **Host**: `{socket.gethostname()}`")
    lines.append(f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"- **Subset size**: {len(subset)}")
    lines.append(f"- **HIP pass rate**: {hip_pass}/{len(subset)}")
    lines.append(f"- **SYCL pass rate**: {sycl_pass}/{len(subset)}")
    lines.append("")
    lines.append("Run times are whatever the per-benchmark regex in `src/scripts/benchmarks/subset.json` captures (kernel time or offload time, mean of repeats). The unit is benchmark-specific (some report ms, some s) — both columns for a given benchmark always use the **same** unit, so the speedup ratio is meaningful. Speedup = `SYCL_time / HIP_time`; values >1 mean chipStar HIP is faster.")
    lines.append("")
    lines.append("| Benchmark | chipStar HIP | HIP time | SYCL (icpx) | SYCL time | Speedup (SYCL/HIP) |")
    lines.append("|-----------|--------------|----------|-------------|-----------|--------------------|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    lines.append("Legend: ✓ ran, ✗ build = compile failed, ✗ run = compiled but runtime error, ⏭ skipped, — no data.")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
