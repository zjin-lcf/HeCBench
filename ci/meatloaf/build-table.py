#!/usr/bin/env python3
"""
Render a chipStar-HIP-vs-SYCL comparison table from two autohecbench.py runs
and (optionally) compare against a checked-in baseline JSON to flag
regressions and improvements.

Inputs (per backend):
  --hip-csv / --sycl-csv          Per-benchmark CSV from autohecbench.py:
                                   '<bench-name>, <run1>, <run2>, ...'
                                   Times are whatever the per-benchmark regex
                                   in subset.json captures (some ms, some s).
  --hip-summary / --sycl-summary  Per-backend JSON summary:
                                   {'<bench-name>': {'compile': '...', 'run': '...'}}
  --baseline                      (Optional) Checked-in baseline JSON; same
                                   schema this script writes to --baseline-out.

Output:
  --out                  Markdown table written to GITHUB_STEP_SUMMARY.
  --baseline-out         New baseline candidate JSON. Always written. The
                         next CI run can adopt it by overwriting the
                         checked-in baseline file.

Exit code:
  0 if there are no regressions vs the supplied baseline (or if no baseline
    was supplied).
  1 if at least one benchmark regressed in correctness (was passing in
    baseline, now failing) or in performance (slower by more than
    --tolerance-pct).
"""

import argparse
import csv
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TOLERANCE_PCT = 15.0  # %

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_csv(path: Path) -> dict:
    """Return {bench_name: mean_time} (autohecbench writes the regex match)."""
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.reader(f):
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


def load_baseline(path):
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Result reduction
# ---------------------------------------------------------------------------

def status_for(summary: dict, key: str):
    """Return ('pass'|'fail-build'|'fail-run'|'absent', raw_dict)."""
    entry = summary.get(key)
    if not entry:
        return "absent", None
    compile_ = entry.get("compile", "")
    run = entry.get("run", "")
    if compile_ != "success":
        return "fail-build", entry
    if run == "success":
        return "pass", entry
    if run == "skipped":
        return "absent", entry
    return "fail-run", entry


STATUS_GLYPH = {
    "pass": "✓",
    "fail-build": "✗ build",
    "fail-run": "✗ run",
    "absent": "—",
}


def reduce_backend(csv_data: dict, summary: dict, suffix: str, base_names):
    """{base_name: {'status': str, 'time': float|None}}"""
    out = {}
    for base in base_names:
        key = f"{base}-{suffix}"
        st, _ = status_for(summary, key)
        time = csv_data.get(key)
        if st == "pass" and time is None:
            # autohecbench summary says success but no CSV row -- treat as fail-run
            st = "fail-run"
        out[base] = {"status": st, "time": time}
    return out


def fmt_time(value):
    if value is None:
        return "—"
    if value < 1.0:
        return f"{value:.4f}"
    if value < 100:
        return f"{value:.3f}"
    return f"{value:.1f}"


def fmt_pct(value):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


# ---------------------------------------------------------------------------
# Diff against baseline
# ---------------------------------------------------------------------------

def diff_backend(base_entry, cur_entry, tolerance_pct):
    """
    Return ('regression'|'improvement'|'unchanged'|'first-seen', detail_str).
    Compare a single (baseline, current) pair for one (benchmark, backend).
    """
    if base_entry is None:
        return "first-seen", "no baseline"
    base_status = base_entry.get("status")
    cur_status = cur_entry["status"]

    # Correctness regression / improvement.
    if base_status == "pass" and cur_status != "pass":
        return "regression", f"was pass, now {cur_status}"
    if base_status != "pass" and cur_status == "pass":
        return "improvement", f"was {base_status}, now pass"

    # Both fail or both absent: unchanged correctness.
    if cur_status != "pass":
        return "unchanged", cur_status

    # Both pass -- compare timings.
    base_t = base_entry.get("time")
    cur_t = cur_entry["time"]
    if base_t is None or cur_t is None or base_t <= 0:
        return "unchanged", "no timing"
    delta_pct = (cur_t - base_t) / base_t * 100.0
    if delta_pct > tolerance_pct:
        return "regression", f"slower by {fmt_pct(delta_pct)} ({fmt_time(base_t)}→{fmt_time(cur_t)})"
    if delta_pct < -tolerance_pct:
        return "improvement", f"faster by {fmt_pct(delta_pct)} ({fmt_time(base_t)}→{fmt_time(cur_t)})"
    return "unchanged", fmt_pct(delta_pct)


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def build_baseline_dict(hip_red, sycl_red, tolerance_pct):
    out = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "perf_tolerance_pct": tolerance_pct,
        "benchmarks": {},
    }
    base_names = sorted(set(hip_red.keys()) | set(sycl_red.keys()))
    for name in base_names:
        out["benchmarks"][name] = {
            "hip": hip_red.get(name, {"status": "absent", "time": None}),
            "sycl": sycl_red.get(name, {"status": "absent", "time": None}),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip-csv", required=True, type=Path)
    ap.add_argument("--hip-summary", required=True, type=Path)
    ap.add_argument("--sycl-csv", required=True, type=Path)
    ap.add_argument("--sycl-summary", required=True, type=Path)
    ap.add_argument("--baseline", type=Path,
                    help="Checked-in baseline JSON to compare against (optional)")
    ap.add_argument("--baseline-out", type=Path, required=True,
                    help="Path to write the new baseline candidate JSON")
    ap.add_argument("--out", type=Path, required=True,
                    help="Markdown report to append to GITHUB_STEP_SUMMARY")
    ap.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT,
                    help="Per-benchmark perf delta tolerance, in percent")
    args = ap.parse_args()

    hip_csv = parse_csv(args.hip_csv)
    sycl_csv = parse_csv(args.sycl_csv)
    hip_sum = parse_summary(args.hip_summary)
    sycl_sum = parse_summary(args.sycl_summary)

    base_names = set()
    for s in (hip_sum, sycl_sum):
        for k in s.keys():
            for suffix in ("-hip", "-sycl"):
                if k.endswith(suffix):
                    base_names.add(k[: -len(suffix)])
                    break
    base_names = sorted(base_names)

    hip_red = reduce_backend(hip_csv, hip_sum, "hip", base_names)
    sycl_red = reduce_backend(sycl_csv, sycl_sum, "sycl", base_names)

    # Always write the new baseline candidate.
    new_baseline = build_baseline_dict(hip_red, sycl_red, args.tolerance_pct)
    args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_out.write_text(json.dumps(new_baseline, indent=2, sort_keys=True) + "\n")

    baseline = load_baseline(args.baseline)
    has_baseline = baseline is not None
    base_benches = baseline.get("benchmarks", {}) if has_baseline else {}
    tol = baseline.get("perf_tolerance_pct", args.tolerance_pct) if has_baseline else args.tolerance_pct

    # Compute per-benchmark diffs.
    regressions = []   # (bench, backend, detail)
    improvements = []  # (bench, backend, detail)
    for name in base_names:
        for backend, cur_red in (("hip", hip_red[name]), ("sycl", sycl_red[name])):
            base_entry = (base_benches.get(name) or {}).get(backend)
            kind, detail = diff_backend(base_entry, cur_red, tol)
            if kind == "regression":
                regressions.append((name, backend, detail))
            elif kind == "improvement":
                improvements.append((name, backend, detail))

    # Tallies.
    def count_pass(red):
        return sum(1 for v in red.values() if v["status"] == "pass")
    hip_pass = count_pass(hip_red)
    sycl_pass = count_pass(sycl_red)
    total = len(base_names)

    # ---------------- Markdown rendering ----------------
    L = []
    L.append("# chipStar HIP vs SYCL — meatloaf CI")
    L.append("")
    L.append(f"- **Host**: `{socket.gethostname()}`")
    L.append(f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    L.append(f"- **Benchmarks**: {total}")
    L.append(f"- **HIP pass rate**:  {hip_pass}/{total}  ({100*hip_pass/max(total,1):.1f}%)")
    L.append(f"- **SYCL pass rate**: {sycl_pass}/{total}  ({100*sycl_pass/max(total,1):.1f}%)")
    if has_baseline:
        L.append(f"- **Baseline**: `{args.baseline}` (generated {baseline.get('generated_at','?')}, tolerance ±{tol:.1f}%)")
    else:
        L.append(f"- **Baseline**: _none — this run will become the first baseline candidate_")
    L.append("")

    if has_baseline:
        if regressions:
            L.append(f"## 🚨 Regressions ({len(regressions)})")
            L.append("")
            L.append("| Benchmark | Backend | Change |")
            L.append("|-----------|---------|--------|")
            for bench, backend, detail in regressions:
                L.append(f"| {bench} | {backend} | {detail} |")
            L.append("")
        else:
            L.append("## 🚨 Regressions")
            L.append("")
            L.append("_None._ Every benchmark that passed in baseline still passes, and no timing degraded by more than the tolerance.")
            L.append("")

        if improvements:
            L.append(f"## 📈 Improvements ({len(improvements)})")
            L.append("")
            L.append("| Benchmark | Backend | Change |")
            L.append("|-----------|---------|--------|")
            for bench, backend, detail in improvements:
                L.append(f"| {bench} | {backend} | {detail} |")
            L.append("")

    # Full table.
    L.append("## Full results")
    L.append("")
    L.append("Run times come from the per-benchmark regex in `src/scripts/benchmarks/subset.json` (mean of repeats). Units are benchmark-specific (some report ms, some s) — both columns for a given row always use the **same** unit, so the speedup ratio is meaningful. Speedup = `SYCL_time / HIP_time`; values >1 mean chipStar HIP is faster.")
    L.append("")
    L.append("| Benchmark | chipStar HIP | HIP time | SYCL (icpx) | SYCL time | Speedup (SYCL/HIP) |")
    L.append("|-----------|--------------|----------|-------------|-----------|--------------------|")
    for name in base_names:
        h = hip_red[name]
        s = sycl_red[name]
        h_glyph = STATUS_GLYPH.get(h["status"], "?")
        s_glyph = STATUS_GLYPH.get(s["status"], "?")
        h_t = fmt_time(h["time"])
        s_t = fmt_time(s["time"])
        if h["time"] and s["time"] and h["time"] > 0:
            speedup = f"{s['time']/h['time']:.2f}x"
        else:
            speedup = "—"
        L.append(f"| {name} | {h_glyph} | {h_t} | {s_glyph} | {s_t} | {speedup} |")
    L.append("")
    L.append("Legend: ✓ ran, ✗ build = compile failed, ✗ run = compiled but runtime error, — no data.")
    L.append("")
    L.append("To accept this run as the new baseline, copy `ci/meatloaf/results/baseline-candidate.json` over `ci/meatloaf/baseline.json` and commit.")
    L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.baseline_out}")
    print(f"Regressions: {len(regressions)}  Improvements: {len(improvements)}")

    return 1 if (has_baseline and regressions) else 0


if __name__ == "__main__":
    sys.exit(main())
