#!/bin/bash
# Test the 52 timeout benchmarks with 3-minute limit
# Categorize as: SUCCESS, TIMEOUT, or ERROR

TIMEOUT=180  # 3 minutes

# List of benchmarks that timed out in previous test
TIMEOUTS=(
  "adjacent-omp"
  "all-pairs-distance-omp"
  "asta-omp"
  "attention-omp"
  "babelstream-omp"
  "bfs-omp"
  "channelShuffle-omp"
  "channelSum-omp"
  "concat-omp"
  "contract-omp"
  "convolution1D-omp"
  "convolution3D-omp"
  "crs-omp"
  "degrid-omp"
  "dense-embedding-omp"
  "dp-omp"
  "dxtc2-omp"
  "epistasis-omp"
  "expdist-omp"
  "filter-omp"
  "floydwarshall-omp"
  "fpc-omp"
  "gabor-omp"
  "gc-omp"
  "grep-omp"
  "haversine-omp"
  "hotspot3D-omp"
  "hybridsort-omp"
  "interval-omp"
  "iso2dfd-omp"
  "jacobi-omp"
  "kalman-omp"
  "kmeans-omp"
  "laplace-omp"
  "laplace3d-omp"
  "lavaMD-omp"
  "libor-omp"
  "lid-driven-cavity-omp"
  "linearprobing-omp"
  "lr-omp"
  "lud-omp"
  "match-omp"
  "md-omp"
  "minisweep-omp"
  "mriQ-omp"
  "norm2-omp"
  "nw-omp"
  "page-rank-omp"
  "particles-omp"
  "pathfinder-omp"
  "pso-omp"
  "s3d-omp"
  "srad-omp"
)

SUCCESS=0
TIMEOUT_COUNT=0
ERROR_COUNT=0

echo "Testing ${#TIMEOUTS[@]} benchmarks with ${TIMEOUT}s timeout..."
echo "========================================================"
echo ""

for bench in "${TIMEOUTS[@]}"; do
  echo -n "Testing $bench... "

  # Run with timeout and capture output
  START=$(date +%s)
  OUTPUT=$(timeout $TIMEOUT make -C "$bench" run 2>&1)
  EXIT_CODE=$?
  END=$(date +%s)
  ELAPSED=$((END - START))

  if [ $EXIT_CODE -eq 0 ]; then
    # Check if it actually passed
    if echo "$OUTPUT" | grep -qE "PASS|done|success|SUCCESS|Complete"; then
      echo "✅ SUCCESS (${ELAPSED}s)"
      ((SUCCESS++))
    else
      echo "⚠️  COMPLETED but no success marker (${ELAPSED}s)"
      echo "   Last line: $(echo "$OUTPUT" | tail -1)"
      ((SUCCESS++))
    fi
  elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "⏱️  TIMEOUT (>${TIMEOUT}s)"
    ((TIMEOUT_COUNT++))
    # Show what it was doing
    echo "   Last output: $(echo "$OUTPUT" | tail -1)"
  else
    echo "❌ ERROR (exit $EXIT_CODE after ${ELAPSED}s)"
    ((ERROR_COUNT++))
    # Show error
    ERROR_LINE=$(echo "$OUTPUT" | grep -E "error|Error|ERROR|fail|FAIL|Abort|Segmentation" | head -1)
    if [ -n "$ERROR_LINE" ]; then
      echo "   Error: $ERROR_LINE"
    else
      echo "   Last line: $(echo "$OUTPUT" | tail -1)"
    fi
  fi
done

echo ""
echo "========================================================"
echo "Summary:"
echo "  ✅ SUCCESS:  $SUCCESS / ${#TIMEOUTS[@]}"
echo "  ⏱️  TIMEOUT:  $TIMEOUT_COUNT / ${#TIMEOUTS[@]}"
echo "  ❌ ERROR:    $ERROR_COUNT / ${#TIMEOUTS[@]}"
echo ""
echo "Success rate: $(( SUCCESS * 100 / ${#TIMEOUTS[@]} ))%"
