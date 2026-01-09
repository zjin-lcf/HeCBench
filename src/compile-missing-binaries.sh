#!/bin/bash
# Try to compile benchmarks without binaries

BENCHES=(
  "amgmk-omp"
  "ans-omp"
  "b+tree-omp"
  "face-omp"
  "gc-omp"
  "lsqt-omp"
  "md5hash-omp"
  "miniFE-omp"
  "miniWeather-omp"
  "multimaterial-omp"
  "myocyte-omp"
  "projectile-omp"
  "qtclustering-omp"
  "sobol-omp"
  "stencil1d-omp"
  "xsbench-omp"
)

echo "Compiling ${#BENCHES[@]} benchmarks without binaries..."
echo "======================================================"
echo ""

COMPILED=0
FAILED=0
TIMEOUT=0

for bench in "${BENCHES[@]}"; do
  echo -n "$bench: "
  
  # Clean first
  make -C "$bench" clean >/dev/null 2>&1
  
  # Try to compile with 60s timeout
  timeout 60 make -C "$bench" >/tmp/compile_${bench}.log 2>&1
  EXIT_CODE=$?
  
  # Check for various binary names
  if [ -f "$bench/main" ] || [ -f "$bench/${bench%-omp}" ] || [ -f "$bench/$(basename $bench -omp)" ]; then
    echo "✅ SUCCESS"
    ((COMPILED++))
  elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "⏱️  TIMEOUT (>60s)"
    ((TIMEOUT++))
  else
    echo "❌ FAILED"
    # Show first error
    grep -m 1 -E "error:|Error|fatal" /tmp/compile_${bench}.log | sed 's/^/    /'
    ((FAILED++))
  fi
done

echo ""
echo "======================================================"
echo "Results:"
echo "  ✅ Compiled: $COMPILED"
echo "  ❌ Failed:   $FAILED"
echo "  ⏱️ Timeout:  $TIMEOUT"
