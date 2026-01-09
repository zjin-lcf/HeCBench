#!/bin/bash
# Compare CUDA vs OMP Makefile run targets
# Check if they use the same arguments and parameters

echo "Comparing CUDA and OMP Makefiles for timeout benchmarks"
echo "========================================================"
echo ""

TIMEOUTS=(
  "adjacent"
  "attention"
  "babelstream"
  "channelShuffle"
  "channelSum"
  "concat"
  "convolution1D"
  "convolution3D"
  "jacobi"
  "kmeans"
  "laplace"
  "lavaMD"
  "nw"
  "pathfinder"
)

for base in "${TIMEOUTS[@]}"; do
  CUDA_DIR="${base}-cuda"
  OMP_DIR="${base}-omp"

  echo "=== $base ==="

  # Check if both exist
  if [ ! -d "$CUDA_DIR" ]; then
    echo "  ⚠️  No CUDA version"
    echo ""
    continue
  fi
  if [ ! -d "$OMP_DIR" ]; then
    echo "  ⚠️  No OMP version"
    echo ""
    continue
  fi

  # Extract run targets
  CUDA_RUN=$(grep -A 1 "^run:" "$CUDA_DIR/Makefile" 2>/dev/null | tail -1 | sed 's/.*\.\///')
  OMP_RUN=$(grep -A 1 "^run:" "$OMP_DIR/Makefile" 2>/dev/null | tail -1 | sed 's/.*\.\///')

  if [ -z "$CUDA_RUN" ]; then
    echo "  ⚠️  No run target in CUDA Makefile"
  fi
  if [ -z "$OMP_RUN" ]; then
    echo "  ⚠️  No run target in OMP Makefile"
  fi

  if [ -n "$CUDA_RUN" ] && [ -n "$OMP_RUN" ]; then
    echo "  CUDA: $CUDA_RUN"
    echo "  OMP:  $OMP_RUN"

    # Check if they match
    if [ "$CUDA_RUN" = "$OMP_RUN" ]; then
      echo "  ✅ MATCH"
    else
      echo "  ⚠️  DIFFERENT!"
    fi

    # Try to actually run CUDA version to see how long it takes
    if [ -f "$CUDA_DIR/main" ] || [ -f "$CUDA_DIR/$(basename $(echo $CUDA_RUN | awk '{print $1}'))" ]; then
      echo -n "  Testing CUDA runtime... "
      START=$(date +%s)
      timeout 30 make -C "$CUDA_DIR" run </dev/null >/dev/null 2>&1
      EXIT_CODE=$?
      END=$(date +%s)
      ELAPSED=$((END - START))

      if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ ${ELAPSED}s"
      elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
        echo "⏱️  timeout (>${ELAPSED}s)"
      else
        echo "❌ error"
      fi
    fi
  fi

  echo ""
done
