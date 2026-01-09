#!/bin/bash
# Test all benchmarks that have compile issues

# From timeout analysis
TIMEOUT_COMPILE=(
  "gc-omp"
  "grep-omp"
  "hybridsort-omp"
  "kmeans-omp"
  "srad-omp"
)

# Check all benchmarks for compile issues
echo "Testing compile issues..."
echo "========================"
echo ""

for bench in gc-omp grep-omp hybridsort-omp kmeans-omp srad-omp; do
  if [ ! -d "$bench" ]; then
    echo "$bench: ❌ Directory doesn't exist"
    continue
  fi
  
  echo "=== $bench ==="
  
  # Clean first
  make -C "$bench" clean >/dev/null 2>&1
  
  # Try to compile with 60s timeout
  timeout 60 make -C "$bench" 2>&1 | tee /tmp/compile_${bench}.log | tail -20
  
  if [ -f "$bench/main" ]; then
    echo "✅ Compiled successfully"
  elif [ ${PIPESTATUS[0]} -eq 124 ] || [ ${PIPESTATUS[0]} -eq 143 ]; then
    echo "⏱️  COMPILE TIMEOUT (>60s)"
  else
    echo "❌ COMPILE FAILED"
    echo "Error summary:"
    grep -iE "error:|fatal|undefined" /tmp/compile_${bench}.log | head -5
  fi
  echo ""
done
