# SHMEMBENCH-OMP Fix Attempts

## Problem
The `shmembench-omp` benchmark uses nested `#pragma omp parallel` with barriers inside `#pragma omp target teams`, which causes nvc++ 25.11 to hang during compilation.

## Approaches Attempted

### 1. Remove Nested Parallel Pragma
- **Tried:** Commented out `#pragma omp parallel`, using only `thread_limit`
- **Result:** FAILED - Compiler hangs indefinitely (timeout after 60+ seconds)
- **Reason:** Even without the nested pragma, the barriers in team context cause compilation hang

### 2. Lower Optimization Level
- **Tried:** Set -O0 instead of -O3
- **Result:** FAILED - Compiler still hangs (timeout after 120+ seconds)
- **Reason:** Not an optimization issue, but a structural limitation with barriers

### 3. Different Pragma Combinations
- **Tried:** Various combinations like `teams distribute`, `teams parallel`, etc.
- **Result:** NOT COMPLETED - Would require removing barriers which breaks algorithm

## Root Cause

The kernel structure requires:
1. Team-local shared memory (`shm_buffer[BLOCK_SIZE*6]`)
2. Synchronization barriers after initialization and between swap phases
3. All threads in a team must participate in barriers

This pattern maps directly to CUDA shared memory + `__syncthreads()`, but OpenMP doesn't have an equivalent that nvc++ can compile.

## Why This Is Hard to Fix

Unlike fsm-omp which had complex control flow, shmembench has a simpler structure but relies on:
- **Shared memory semantics:** Array allocated at team scope, accessed by all threads
- **Barrier synchronization:** Required for correctness of the memory benchmark
- **Performance measurement:** The whole point is to measure shared memory bandwidth

Any workaround that removes barriers or uses global memory would:
1. Change what's being measured
2. Give incorrect performance results
3. Defeat the purpose of the benchmark

## Recommendation

**Status:** Cannot fix without changing benchmark semantics

This benchmark fundamentally requires:
- Shared memory visible to all threads in a team
- Synchronization barriers
- These features together in OpenMP offload

**Alternatives:**
1. Use the CUDA version (`shmembench-cuda`) which works correctly
2. Wait for improved nvc++ support for barriers in team constructs
3. Complete algorithmic rewrite to avoid barriers (but this changes what's measured)

The shmembench-cuda version should be used for actual shared memory performance measurements.
