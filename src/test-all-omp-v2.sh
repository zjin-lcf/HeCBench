#!/bin/bash

# Script to compile and run all -omp benchmarks using 'make run'
# This version uses make run to automatically get the correct arguments
# Usage: ./test-all-omp-v2.sh

RESULTS_DIR="omp-test-results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_DIR/test-run-$TIMESTAMP.log"
SUMMARY_FILE="$RESULTS_DIR/summary-$TIMESTAMP.txt"

SUCCESS_COUNT=0
COMPILE_FAIL_COUNT=0
RUN_FAIL_COUNT=0

echo "=== OMP Benchmark Testing Started at $(date) ===" | tee "$LOG_FILE"
echo "Using 'make run' to execute benchmarks with proper arguments" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Find all -omp directories
BENCHMARKS=($(find . -maxdepth 1 -type d -name "*-omp" | sort))
TOTAL=${#BENCHMARKS[@]}

echo "Found $TOTAL -omp benchmarks" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for i in "${!BENCHMARKS[@]}"; do
    BENCH_DIR="${BENCHMARKS[$i]}"
    BENCH_NAME=$(basename "$BENCH_DIR")
    PROGRESS=$((i + 1))

    echo "[$PROGRESS/$TOTAL] Testing $BENCH_NAME..." | tee -a "$LOG_FILE"

    cd "$BENCH_DIR" || continue

    # Clean previous builds
    make clean &>/dev/null

    # Try to compile (with 3 minute timeout)
    echo "  Compiling..." | tee -a "../$LOG_FILE"
    if timeout 180s make -j$(nproc) &> "../$RESULTS_DIR/${BENCH_NAME}_compile.log"; then
        echo "  ✓ Compilation successful" | tee -a "../$LOG_FILE"

        # Check if Makefile has a run target
        if grep -q "^run:" Makefile 2>/dev/null; then
            echo "  Running via 'make run'..." | tee -a "../$LOG_FILE"

            # Run with timeout of 30 seconds using make run
            if timeout 30s make run &> "../$RESULTS_DIR/${BENCH_NAME}_run.log"; then
                echo "  ✓ Run successful" | tee -a "../$LOG_FILE"
                echo "$BENCH_NAME: SUCCESS" >> "../$SUMMARY_FILE"
                ((SUCCESS_COUNT++))
            else
                EXIT_CODE=$?
                echo "  ✗ Run failed (exit code: $EXIT_CODE)" | tee -a "../$LOG_FILE"
                echo "$BENCH_NAME: RUN_FAIL (exit code: $EXIT_CODE)" >> "../$SUMMARY_FILE"
                ((RUN_FAIL_COUNT++))
            fi
        else
            echo "  ✗ No run target in Makefile" | tee -a "../$LOG_FILE"
            echo "$BENCH_NAME: NO_RUN_TARGET" >> "../$SUMMARY_FILE"
            ((RUN_FAIL_COUNT++))
        fi
    else
        EXIT_CODE=$?
        echo "  ✗ Compilation failed (exit code: $EXIT_CODE)" | tee -a "../$LOG_FILE"
        echo "$BENCH_NAME: COMPILE_FAIL (exit code: $EXIT_CODE)" >> "../$SUMMARY_FILE"
        ((COMPILE_FAIL_COUNT++))
    fi

    echo "" | tee -a "../$LOG_FILE"
    cd ..
done

echo "=== Testing Complete at $(date) ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results Summary:" | tee -a "$LOG_FILE"
echo "  Total benchmarks: $TOTAL" | tee -a "$LOG_FILE"
echo "  Successful: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "  Compilation failures: $COMPILE_FAIL_COUNT" | tee -a "$LOG_FILE"
echo "  Runtime failures: $RUN_FAIL_COUNT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Detailed logs saved to: $RESULTS_DIR/" | tee -a "$LOG_FILE"
echo "Summary saved to: $SUMMARY_FILE" | tee -a "$LOG_FILE"

# Create failure lists
grep "COMPILE_FAIL" "$SUMMARY_FILE" | cut -d: -f1 > "$RESULTS_DIR/compile_failures.txt" 2>/dev/null
grep "RUN_FAIL\|NO_RUN_TARGET" "$SUMMARY_FILE" | cut -d: -f1 > "$RESULTS_DIR/run_failures.txt" 2>/dev/null
grep "SUCCESS" "$SUMMARY_FILE" | cut -d: -f1 > "$RESULTS_DIR/successes.txt" 2>/dev/null

echo ""
echo "Failure lists created:"
echo "  Compile failures: $RESULTS_DIR/compile_failures.txt"
echo "  Run failures: $RESULTS_DIR/run_failures.txt"
echo "  Successes: $RESULTS_DIR/successes.txt"
