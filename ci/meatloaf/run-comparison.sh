#!/usr/bin/env bash
#
# meatloaf CI driver: build & run the curated benchmark subset twice on the
# Intel GPU -- once with chipStar's hipcc (HIP-on-SPIR-V) and once with
# oneAPI icpx (SYCL on Level Zero) -- then emit a side-by-side comparison
# table.
#
# Idempotent: safe to re-run; produces everything under ci/meatloaf/results/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS="$REPO_ROOT/ci/meatloaf/results"
SUBSET_FILE="$REPO_ROOT/ci/meatloaf/benchmarks-subset.txt"

mkdir -p "$RESULTS"
rm -f "$RESULTS"/*.csv "$RESULTS"/*.json "$RESULTS"/comparison.md

# ---------------------------------------------------------------------------
# Module environment
# ---------------------------------------------------------------------------
# `set -u` plus `module` shell functions interact badly under bash; toggle off
# nounset around the module commands.
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

# Read the curated benchmark list (one base name per line, '#' comments OK).
SUBSET=()
while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | tr -d '[:space:]')"
    [ -n "$line" ] && SUBSET+=("$line")
done < "$SUBSET_FILE"

if [ ${#SUBSET[@]} -eq 0 ]; then
    echo "ERROR: $SUBSET_FILE is empty" >&2
    exit 1
fi
echo "Subset: ${SUBSET[*]}"
echo

# autohecbench.py expects per-benchmark targets like 'ace-hip', 'ace-sycl'.
HIP_TARGETS=()
SYCL_TARGETS=()
for b in "${SUBSET[@]}"; do
    HIP_TARGETS+=("${b}-hip")
    SYCL_TARGETS+=("${b}-sycl")
done

AUTO="$REPO_ROOT/src/scripts/autohecbench.py"

# ---------------------------------------------------------------------------
# Pass 1: chipStar HIP
# ---------------------------------------------------------------------------
echo "=== Pass 1: chipStar HIP ==="
python3 "$AUTO" \
    --yes-prompt \
    --clean \
    --compiler-name hipcc \
    --output  "$RESULTS/hip-chipstar.csv" \
    --summary "$RESULTS/hip-chipstar.json" \
    "${HIP_TARGETS[@]}" || echo "(autohecbench HIP pass exited non-zero -- continuing so the table still renders)"

# ---------------------------------------------------------------------------
# Pass 2: SYCL on Intel GPU (Level Zero via icpx, no CUDA / no HIP backend)
# ---------------------------------------------------------------------------
echo "=== Pass 2: SYCL (icpx) ==="
python3 "$AUTO" \
    --yes-prompt \
    --clean \
    --compiler-name icpx \
    --sycl-type opencl \
    --output  "$RESULTS/sycl.csv" \
    --summary "$RESULTS/sycl.json" \
    "${SYCL_TARGETS[@]}" || echo "(autohecbench SYCL pass exited non-zero -- continuing so the table still renders)"

# ---------------------------------------------------------------------------
# Build the comparison table
# ---------------------------------------------------------------------------
echo "=== Building comparison table ==="
python3 "$REPO_ROOT/ci/meatloaf/build-table.py" \
    --hip-csv      "$RESULTS/hip-chipstar.csv" \
    --hip-summary  "$RESULTS/hip-chipstar.json" \
    --sycl-csv     "$RESULTS/sycl.csv" \
    --sycl-summary "$RESULTS/sycl.json" \
    --subset       "$SUBSET_FILE" \
    --out          "$RESULTS/comparison.md"

echo
echo "=== comparison.md ==="
cat "$RESULTS/comparison.md"
