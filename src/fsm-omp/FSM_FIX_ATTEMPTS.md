# FSM-OMP Fix Attempts

## Problem
The `fsm-omp` benchmark uses nested `#pragma omp parallel` inside `#pragma omp target teams`, which nvc++ 25.11 cannot compile due to complex control flow with do-while loops and barriers.

**Error:** `Missing branch target block`

## Approaches Attempted

### 1. Lower Optimization Levels
- **Tried:** -O3, -O1, -O0
- **Result:** FAILED - Same error at all optimization levels
- **Reason:** Not an optimizer bug, but a structural limitation

### 2. Remove Nested Parallel Pragma
- **Tried:** Commented out `#pragma omp parallel`, relying only on `thread_limit`
- **Result:** FAILED - Same "Missing branch target block" error
- **Reason:** The complex control flow (do-while with barriers) is incompatible with target teams

### 3. Flatten to Teams Distribute Parallel For
- **Tried:** `#pragma omp target teams distribute parallel for` with single iteration
- **Result:** FAILED - "Illegal context for barrier" errors
- **Reason:** Barriers not allowed in `distribute parallel for` construct

### 4. CPU Multicore Mode
- **Tried:** Compile with `-mp=multicore` instead of `-mp=gpu`
- **Result:** FAILED - Different LLVM error: `use of undefined value`
- **Reason:** Control flow still too complex even for CPU OpenMP

### 5. Teams + Distribute Without Parallel
- **Tried:** `#pragma omp target teams num_teams(POPCNT)` with `#pragma omp distribute`
- **Result:** NOT COMPLETED - Would require major rewrite of team_next array allocation

## Root Cause

The fundamental issue is the kernel structure:
1. Nested parallel region inside target teams
2. Do-while loop with condition based on shared state (`same[bid] < CUTOFF`)
3. Multiple barriers inside the loop
4. Complex shared memory access patterns

This combination creates control flow that nvc++ cannot translate to GPU code.

## Possible Solutions (Requiring Major Rewrite)

1. **Split into multiple kernels:** Break the do-while loop into separate kernel launches
2. **Restructure algorithm:** Remove do-while, use fixed iteration count
3. **Use CUDA directly:** This algorithm may not be suitable for OpenMP offloading

## Recommendation

**Status:** Cannot fix without major algorithmic rewrite

This benchmark requires either:
- Complete kernel restructuring to avoid nested parallelism
- Using CUDA/HIP version instead of OpenMP
- Waiting for improved nvc++ compiler support for complex nested constructs

The fsm-cuda version should work fine as it's designed for the CUDA programming model.
