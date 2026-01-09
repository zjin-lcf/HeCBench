#!/bin/bash
# Test the 13 newly compiled benchmarks

BENCHMARKS=(
  "amgmk-omp:AMGMk"
  "ans-omp:bin/main"
  "b+tree-omp:b+tree.out"
  "face-omp:vj-gpu"
  "lsqt-omp:lsqt_gpu"
  "md5hash-omp:MD5Hash"
  "multimaterial-omp:multimat"
  "myocyte-omp:myocyte.out"
  "projectile-omp:Projectile"
  "qtclustering-omp:qtc"
  "sobol-omp:SobolQRNG"
  "stencil1d-omp:stencil_1d"
  "xsbench-omp:XSBench"
)

echo "Testing 13 newly compiled benchmarks..."
echo "========================================"
echo ""

SUCCESS=0
FAIL=0

for item in "${BENCHMARKS[@]}"; do
  bench="${item%:*}"
  binary="${item#*:}"
  
  echo -n "$bench: "
  
  # Run with 60s timeout
  timeout 60 make -C "$bench" run </dev/null >/tmp/test_${bench}.out 2>&1
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    # Check for PASS or success
    if grep -qiE "PASS|SUCCESS|correct|verification" /tmp/test_${bench}.out 2>/dev/null; then
      echo "✅ PASS"
      ((SUCCESS++))
    else
      echo "⚠️  Completed"
      tail -2 /tmp/test_${bench}.out | head -1
      ((SUCCESS++))
    fi
  elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "⏱️  TIMEOUT"
    ((FAIL++))
  else
    echo "❌ FAILED (exit $EXIT_CODE)"
    grep -iE "error|fail|cannot" /tmp/test_${bench}.out | head -1
    ((FAIL++))
  fi
done

echo ""
echo "========================================"
echo "Results:"
echo "  ✅ Success: $SUCCESS"
echo "  ❌ Failed:  $FAIL"
