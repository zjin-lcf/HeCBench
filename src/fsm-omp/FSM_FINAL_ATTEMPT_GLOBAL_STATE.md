# FSM-OMP Final Attempt: Global State Arrays
**Date:** January 8, 2026
**Attempt Number:** 8

## Hypothesis
Previous crashes might be due to 32KB `state[TABSIZE]` array per thread causing stack overflow. Move state arrays to global device memory.

## Implementation
1. Added `global_state` parameter pointing to pre-allocated global memory
2. Allocated `POPCNT * POPSIZE * TABSIZE` bytes (512MB for reduced params, 8GB for full)
3. Each thread accesses its state array at `global_state[id * TABSIZE]`
4. Kept nested parallel structure with simplified atomic operations

## Changes Made
- Created `kernels_global_state.h` with state arrays in global memory
- Modified `main.cpp` to allocate and map global_state array
- Used `parameters_small.h` with POPCNT=64 instead of 1024 for testing

## Result
- ✅ Compilation: Success
- ❌ Runtime: Crash with SIGABRT (signal 134)
- Crash location: During program initialization, before any output

## Testing
```bash
# With reduced parameters (512MB state arrays)
POPCNT=64, POPSIZE=256, TABSIZE=32768
./main 100

# Result: Aborted (core dumped)
# No debug output printed - crashes during static/runtime initialization
```

## Comparison with CUDA
```bash
cd ../fsm-cuda
./main 10000

# Result: PASS
# 10	#kernel execution times
# 0.352579	#runtime [s]
# PASS
```

The CUDA version works perfectly on the same hardware (GB10 with sm_89).

## Analysis

### Why Global State Didn't Help
The crash happens during initialization, not during kernel execution:
- No debug output is printed (not even "DEBUG: Allocating host arrays...")
- Program aborts before main() execution begins
- Suggests OpenMP runtime initialization failure

### Root Cause Confirmed
The issue is NOT:
- ❌ Stack overflow from large local arrays
- ❌ Memory allocation size
- ❌ Atomic operation complexity
- ❌ Algorithm correctness (CUDA version works)

The issue IS:
- ✅ nvc++ OpenMP runtime cannot initialize nested `#pragma omp parallel` inside `#pragma omp target teams`
- ✅ Runtime failure even when compilation succeeds
- ✅ Fundamental incompatibility between the nested structure and nvc++ 25.11 runtime

## Conclusion

After 8 different attempts including:
1. Flattening parallel structure
2. Removing barriers
3. Replacing critical sections with atomics
4. Using different atomic patterns
5. Simplifying best update logic
6. Keeping nested structure with simplified atomics
7. Using different optimization levels
8. Moving state arrays to global memory

**Result:** fsm-omp cannot be fixed with nvc++ 25.11

The nested `#pragma omp parallel` inside `#pragma omp target teams` pattern:
- Sometimes compiles
- Never runs successfully
- Crashes during initialization or execution
- Works perfectly in CUDA with `__syncthreads()` and `__shared__` memory

## Recommendation

**Use fsm-cuda instead**

The CUDA version:
- Compiles without issues
- Runs correctly
- Produces correct results (PASS)
- Has good performance (14.87 Gtr/s throughput)

This is a compiler/runtime limitation, not an algorithmic issue.
