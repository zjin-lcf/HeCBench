#!/bin/bash

# Final comprehensive Makefile fixer
# Fixes all remaining flag compatibility issues for nvc++

echo "Applying final fixes to all -omp Makefiles..."

# Fix remaining -qnextgen flags (with various spacing)
find . -maxdepth 2 -name "Makefile" -path "*-omp/*" -exec sed -i 's/-qnextgen//g' {} \;

# Fix remaining -fopenmp flags in += patterns
find . -maxdepth 2 -name "Makefile" -path "*-omp/*" -exec sed -i 's/+=-fopenmp$/+=-mp=multicore/g' {} \;

echo "Done! Fixed all remaining compatibility issues."
