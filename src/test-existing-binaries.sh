#!/bin/bash
# Test timeout benchmarks that already have binaries
# Quick classification: FAST (<10s), MEDIUM (10-60s), SLOW (>60s), ERROR

TIMEOUTS=(
  "adjacent-omp"
  "attention-omp"
  "babelstream-omp"
  "channelShuffle-omp"
  "channelSum-omp"
  "concat-omp"
  "convolution1D-omp"
  "convolution3D-omp"
  "jacobi-omp"
  "kmeans-omp"
  "laplace-omp"
  "lavaMD-omp"
  "nw-omp"
  "pathfinder-omp"
)

FAST=0
MEDIUM=0
SLOW=0
ERROR=0

echo "Testing benchmarks with existing binaries..."
echo "================================================"
echo ""

for bench in "${TIMEOUTS[@]}"; do
  if [ ! -f "$bench/main" ]; then
    continue
  fi
  
  echo -n "Testing $bench... "
  
  START=$(date +%s)
  timeout 90 make -C "$bench" run </dev/null >/dev/null 2>&1
  EXIT_CODE=$?
  END=$(date +%s)
  ELAPSED=$((END - START))
  
  if [ $EXIT_CODE -eq 0 ]; then
    if [ $ELAPSED -lt 10 ]; then
      echo "✅ FAST (${ELAPSED}s)"
      ((FAST++))
    elif [ $ELAPSED -lt 60 ]; then
      echo "✅ MEDIUM (${ELAPSED}s)"
      ((MEDIUM++))
    else
      echo "✅ SLOW (${ELAPSED}s)"
      ((SLOW++))
    fi
  elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "⏱️  VERY SLOW (>90s)"
    ((SLOW++))
  else
    echo "❌ ERROR (exit $EXIT_CODE)"
    ((ERROR++))
  fi
done

echo ""
echo "================================================"
echo "Results:"
echo "  ✅ FAST (<10s):     $FAST"
echo "  ✅ MEDIUM (10-60s): $MEDIUM"
echo "  ✅ SLOW (>60s):     $SLOW"
echo "  ❌ ERROR:           $ERROR"
echo ""
echo "Total working: $((FAST + MEDIUM + SLOW))"
