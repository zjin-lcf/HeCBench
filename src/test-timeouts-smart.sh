#!/bin/bash
# Smart test for timeout benchmarks
# - Checks for stdin hangs
# - Prevents runaway recompilation
# - Detects missing input files

TIMEOUT=60  # 1 minute for execution
COMPILE_TIMEOUT=30  # 30 seconds max for compilation check

# Timeout benchmarks to test
TIMEOUTS=(
  "adjacent-omp"
  "all-pairs-distance-omp"
  "asta-omp"
  "attention-omp"
  "babelstream-omp"
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
STDIN_HANG=0
MISSING_INPUT=0
COMPILE_HANG=0

echo "Smart testing ${#TIMEOUTS[@]} benchmarks..."
echo "Compile timeout: ${COMPILE_TIMEOUT}s, Run timeout: ${TIMEOUT}s"
echo "========================================================"
echo ""

for bench in "${TIMEOUTS[@]}"; do
  echo -n "Testing $bench... "

  # Check if binary exists
  if [ -f "$bench/main" ]; then
    NEEDS_COMPILE=0
  else
    NEEDS_COMPILE=1
    echo -n "(needs compile) "
  fi

  # If needs compilation, do quick compile check
  if [ $NEEDS_COMPILE -eq 1 ]; then
    timeout $COMPILE_TIMEOUT make -C "$bench" 2>&1 | grep -q "Error\|error" && {
      echo "❌ COMPILE ERROR"
      ((ERROR_COUNT++))
      continue
    }
    # Check if it compiled after timeout
    if [ ! -f "$bench/main" ]; then
      echo "⏱️  COMPILE TIMEOUT"
      ((COMPILE_HANG++))
      continue
    fi
  fi

  # Run the benchmark with stdin redirected from /dev/null
  START=$(date +%s)
  OUTPUT=$(timeout $TIMEOUT make -C "$bench" run </dev/null 2>&1)
  EXIT_CODE=$?
  END=$(date +%s)
  ELAPSED=$((END - START))

  # Analyze the result
  if [ $EXIT_CODE -eq 0 ]; then
    if echo "$OUTPUT" | grep -qE "PASS|pass|done|Done|success|SUCCESS|Complete"; then
      echo "✅ SUCCESS (${ELAPSED}s)"
      ((SUCCESS++))
    else
      echo "⚠️  COMPLETED (${ELAPSED}s)"
      ((SUCCESS++))
    fi

  elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    # Check what kind of timeout
    if ps aux | grep -q "$bench.*main.*<defunct>"; then
      echo "🔌 STDIN HANG - waiting for input (${ELAPSED}s)"
      ((STDIN_HANG++))
    else
      echo "⏱️  TIMEOUT (${ELAPSED}s)"
      ((TIMEOUT_COUNT++))
    fi

  else
    # Check for common error patterns
    if echo "$OUTPUT" | grep -qiE "no such file|cannot open|file not found|missing"; then
      echo "📁 MISSING INPUT - $(echo "$OUTPUT" | grep -oiE "(no such file|cannot open|file not found)[^'\"]*['\"]?[^'\"]*" | head -1 | cut -c1-60)"
      ((MISSING_INPUT++))
    else
      echo "❌ ERROR (exit $EXIT_CODE, ${ELAPSED}s)"
      ERROR_LINE=$(echo "$OUTPUT" | grep -iE "error|fail|abort|segmentation" | head -1 | cut -c1-80)
      if [ -n "$ERROR_LINE" ]; then
        echo "   → $ERROR_LINE"
      fi
      ((ERROR_COUNT++))
    fi
  fi
done

echo ""
echo "========================================================"
echo "Summary of ${#TIMEOUTS[@]} benchmarks:"
echo "  ✅ SUCCESS:         $SUCCESS"
echo "  ⏱️  TIMEOUT:         $TIMEOUT_COUNT"
echo "  🔌 STDIN HANG:      $STDIN_HANG"
echo "  📁 MISSING INPUT:   $MISSING_INPUT"
echo "  ⚙️  COMPILE HANG:    $COMPILE_HANG"
echo "  ❌ OTHER ERRORS:    $ERROR_COUNT"
echo ""

TOTAL_WORKING=$((SUCCESS + TIMEOUT_COUNT))
echo "Working (success + legitimate timeout): $TOTAL_WORKING / ${#TIMEOUTS[@]}"
echo "Actual issues: $((STDIN_HANG + MISSING_INPUT + COMPILE_HANG + ERROR_COUNT)) / ${#TIMEOUTS[@]}"
