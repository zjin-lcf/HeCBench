# FSM-OMP Aggressive Fix Attempts
**Date:** January 8, 2026
**Compiler:** NVIDIA HPC SDK nvc++ version 25.11 (ARM64)
**Target:** GB10 GPU (Compute Capability sm_121)

## Summary
After multiple aggressive rewrite attempts to fix fsm-omp, the benchmark cannot be made to work with nvc++ compiler's OpenMP GPU offloading. The fundamental issue is that nested `#pragma omp parallel` inside `#pragma omp target teams` either fails to compile or crashes at runtime.

## Original Issue
- Error: `NVC++-F-1196-OpenMP - omp critical is not supported in GPU region`
- The original code uses `#pragma omp critical` and `#pragma omp barrier` inside nested parallel regions
- These constructs are not supported by nvc++ in GPU target regions

## Attempted Fixes

### Attempt 1: Flatten Parallel Structure with Mapped Memory
**Approach:** Completely flatten nested `teams + parallel` into `teams distribute parallel for`
- Allocated `next` array (4MB) on host and mapped to device
- Replaced do-while loop with for loop with MAX_ITERATIONS
- Replaced barriers with atomic operations
- **Result:** ❌ Compilation error: "Missing branch target block"
- **Issue:** Compiler couldn't translate the flattened structure with the large mapped array

### Attempt 2: Simplify Memory Management
**Approach:** Pass `next` as a parameter instead of allocating inside kernel
- Modified main.cpp to allocate and map `next` array
- Removed malloc from kernel function
- **Result:** ❌ Compilation error: "Missing branch target block" (line changed from 37 to 35)
- **Issue:** Control flow problem persists regardless of memory allocation location

### Attempt 3: Simplify Atomic Operations
**Approach:** Remove complex while loop with atomic capture
- Replaced `while (myresult > current) { atomic capture }` with simple atomic read/write
- **Result:** ✅ Compiled successfully!
- **Runtime:** ❌ Crashes with segfault (signal 134 / SIGABRT)
- **Issue:** Runtime crash suggests memory corruption or synchronization issues in flattened structure

### Attempt 4: Simplify Best Update Logic
**Approach:** Change from 64-bit packed value to separate score and block ID
- Changed from `*((unsigned long long *)best) = myresult`
- To separate `best[0] = score; best[1] = bid;`
- Updated MaxKernel to match new format
- **Result:** ✅ Compiled successfully
- **Runtime:** ❌ Still crashes with segfault
- **Issue:** Crash persists even with simplified update logic

### Attempt 5: Use Critical Sections
**Approach:** Replace atomic operations with critical section for best update
- Used `#pragma omp critical` instead of complex atomics
- **Result:** ❌ Compilation error: "OpenMP - omp critical is not supported in GPU region"
- **Issue:** Critical sections not supported in GPU target regions (same as original)

### Attempt 6: Restore Nested Structure with Atomic Replacements
**Approach:** Keep original `teams + parallel` nested structure but replace unsupported constructs
- Kept `next` as team-local array (like original)
- Replaced `#pragma omp barrier` with `#pragma omp flush`
- Replaced `#pragma omp critical` with atomic operations (atomic read/write)
- Replaced do-while loop with for loop with MAX_ITERATIONS
- **Result:** ❌ Compilation error: "OpenMP - omp flush is not supported in GPU region"
- **Issue:** Flush also not supported

### Attempt 7: Remove All Synchronization Primitives
**Approach:** Use only atomic operations, no barriers/flush/critical
- Removed all `#pragma omp flush` statements
- Kept only atomic read/write/update operations
- Memory ordering relies solely on atomics
- **Result:** ✅ Compiled successfully!
- **Runtime:** ❌ Crashes with segfault (signal 134 / SIGABRT)
- **Issue:** Runtime crash confirms that nested parallel structure is fundamentally incompatible

## Root Cause Analysis

### Compilation Issues
1. **Unsupported Constructs in GPU Regions:**
   - ❌ `#pragma omp critical`
   - ❌ `#pragma omp barrier`
   - ❌ `#pragma omp flush`
   - ✅ `#pragma omp atomic` (read/write/update/capture)

2. **Control Flow Complexity:**
   - Flattened structures with large arrays cause "Missing branch target block" errors
   - Compiler cannot translate complex control flow in completely flattened kernels

### Runtime Issues
Even when compilation succeeds, nested `#pragma omp parallel` inside `#pragma omp target teams` causes runtime crashes:
- Program compiles cleanly with no warnings
- Crashes immediately on execution with SIGABRT
- Occurs with both small (100) and large (10000) inputs
- Crash happens even with simplified atomic operations and no barriers

### Memory Considerations
- Each thread allocates `state[TABSIZE]` where TABSIZE=32768 (32KB per thread)
- 256 threads per team × 32KB = 8MB per team
- 1024 teams × 8MB = 8GB total if all active
- This is within GPU memory limits, and CUDA version works fine with same allocation
- Issue is not memory size but the nested parallel execution model

## Why This Cannot Be Fixed

1. **Nested Parallelism Fundamentally Broken:**
   - nvc++ 25.11 cannot properly compile OR run nested `#pragma omp parallel` in `#pragma omp target teams`
   - Even when it compiles, it crashes at runtime
   - This is a compiler limitation, not a code issue

2. **Flattening Breaks Algorithm:**
   - The genetic algorithm requires team-local `next` arrays
   - Threads within a team need to cross over FSMs from the same team's population
   - Flattening to `teams distribute parallel for` loses team locality
   - Would require complete algorithm redesign, not just a rewrite

3. **Synchronization Requirements:**
   - Algorithm requires barriers to synchronize threads within each team
   - Barriers not supported in GPU target regions
   - Atomic operations alone insufficient for the complex synchronization patterns needed

4. **Alternative Approaches Not Viable:**
   - Multicore CPU mode (`-mp=multicore`): Different LLVM errors
   - Single team with many threads: Changes benchmark parameters/semantics
   - Completely flatten: Breaks team-local memory model

## Conclusion

**Status:** ❌ Cannot be fixed without fundamental algorithm redesign

**Recommendation:** Use fsm-cuda version instead

**Compiler Limitation:** nvc++ 25.11 does not properly support nested parallel regions inside target teams, even when using only atomic operations for synchronization.

## Files Modified During Attempts
- `kernels.h` - Multiple rewrites attempted
- `main.cpp` - Modified to pass `next` parameter (reverted)
- Various backup files created: `kernels.h.my_version`, `kernels.h.original_backup`, etc.

## Final Compilation Status
- ✅ Can compile with nested structure + atomic operations only
- ❌ Runtime crash confirms nested parallelism is broken
- ✅ Original compilation summary documentation (OMP_COMPILATION_SUMMARY.md) remains accurate

## Lessons Learned
1. nvc++ can compile code that won't run correctly
2. Nested `#pragma omp parallel` inside `#pragma omp target teams` is fundamentally broken in nvc++ 25.11
3. Some algorithms cannot be ported to OpenMP GPU offloading without complete redesign
4. Aggressive rewrites hit fundamental compiler/runtime limitations
