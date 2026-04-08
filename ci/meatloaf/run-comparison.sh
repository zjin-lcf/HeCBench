#!/usr/bin/env bash
#
# meatloaf CI driver: build & run the FULL HeCBench HIP+SYCL set on the
# Intel GPU -- once with chipStar's hipcc (HIP-on-SPIR-V), once with
# oneAPI icpx (SYCL on Level Zero) -- then diff against the checked-in
# ci/meatloaf/baseline.json and emit a regression/improvement report.
#
# Idempotent: safe to re-run; everything lives under ci/meatloaf/results/.
#
# IMPORTANT: We deliberately call autohecbench.py with small batches of
# benchmarks rather than passing the literal `hip` / `sycl` aliases.
# autohecbench's compile phase spawns one Process per benchmark with no
# concurrency limit (see src/scripts/autohecbench.py:244-253), which
# fork-bombs the host when there are 100+ benchmarks (icpx in particular
# is >1GB RSS per invocation, so it swap-thrashes the box into a
# wedged state). Batching keeps peak parallelism bounded.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS="$REPO_ROOT/ci/meatloaf/results"
BASELINE="$REPO_ROOT/ci/meatloaf/baseline.json"
SUBSET_JSON="$REPO_ROOT/src/scripts/benchmarks/subset.json"

# Max concurrent compiles per autohecbench invocation. icpx peaks ~1.5 GB
# RSS, so 8 keeps us under ~12 GB even on a modest box.
BATCH=${BATCH:-8}

mkdir -p "$RESULTS"
rm -f "$RESULTS"/hip-chipstar.csv "$RESULTS"/hip-chipstar.json \
      "$RESULTS"/sycl.csv "$RESULTS"/sycl.json \
      "$RESULTS"/comparison.md "$RESULTS"/baseline-candidate.json

# ---------------------------------------------------------------------------
# Module environment
# ---------------------------------------------------------------------------
# `set -u` plus `module` shell functions interact badly under bash; toggle
# off nounset around the module commands.
set +u
source /etc/profile.d/modules.sh
module use "$HOME/modulefiles"
module purge
module load llvm/22.0-native
module load oneapi/2025.0.4
module load HIP/chipStar/main
module load HIP/H4I-MKLShim/2026.04.08
module load HIP/H4I-HipBLAS/2026.04.08
module load HIP/H4I-HipSOLVER/2026.04.08
module load HIP/H4I-HipFFT/2026.04.08
module load level-zero/dgpu
module load opencl/dgpu
set -u

echo "=== Tool versions ==="
which hipcc icpx
hipcc --version | head -3 || true
icpx --version | head -3 || true
echo

AUTO="$REPO_ROOT/src/scripts/autohecbench.py"

# ---------------------------------------------------------------------------
# Build the list of benchmark base names from subset.json (one per line).
# ---------------------------------------------------------------------------
mapfile -t BASES < <(python3 -c "
import json
with open('$SUBSET_JSON') as f:
    for k in sorted(json.load(f).keys()):
        print(k)
")
echo "Sweeping ${#BASES[@]} benchmarks in batches of $BATCH"

run_pass () {
    local label="$1" suffix="$2"; shift 2
    local csv="$RESULTS/$label.csv"
    local summary="$RESULTS/$label.json"
    local i n=${#BASES[@]}
    echo "=== Pass: $label ==="
    date
    for ((i = 0; i < n; i += BATCH)); do
        local chunk=("${BASES[@]:i:BATCH}")
        local targets=()
        for b in "${chunk[@]}"; do targets+=("${b}-${suffix}"); done
        echo "--- batch $((i/BATCH+1))/$(((n + BATCH - 1) / BATCH)): ${targets[*]}"
        python3 "$AUTO" \
            --yes-prompt \
            "$@" \
            --output  "$csv" \
            --summary "$summary" \
            "${targets[@]}" \
          || echo "(batch exited non-zero -- continuing)"
    done
    date
}

# ---------------------------------------------------------------------------
# Pass 1: chipStar HIP -- full sweep, batched.
# ---------------------------------------------------------------------------
run_pass "hip-chipstar" "hip" \
    --compiler-name hipcc

# ---------------------------------------------------------------------------
# Pass 2: SYCL on Intel GPU (Level Zero via icpx).
# ---------------------------------------------------------------------------
run_pass "sycl" "sycl" \
    --compiler-name icpx \
    --sycl-type opencl

# ---------------------------------------------------------------------------
# Build the comparison table and diff against the baseline.
#
# build-table.py exits 1 if there are correctness/perf regressions vs the
# checked-in baseline -- that's how this job goes red on a regression.
# ---------------------------------------------------------------------------
echo "=== Building comparison table ==="
python3 "$REPO_ROOT/ci/meatloaf/build-table.py" \
    --hip-csv      "$RESULTS/hip-chipstar.csv" \
    --hip-summary  "$RESULTS/hip-chipstar.json" \
    --sycl-csv     "$RESULTS/sycl.csv" \
    --sycl-summary "$RESULTS/sycl.json" \
    --baseline     "$BASELINE" \
    --baseline-out "$RESULTS/baseline-candidate.json" \
    --out          "$RESULTS/comparison.md"
TABLE_STATUS=$?

echo
echo "=== comparison.md ==="
cat "$RESULTS/comparison.md"

exit $TABLE_STATUS
