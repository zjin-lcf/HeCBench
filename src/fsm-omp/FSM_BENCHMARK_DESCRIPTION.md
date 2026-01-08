# FSM Benchmark: Detailed Description
**Full Name:** FSM_GA - Finite State Machine Genetic Algorithm
**Author:** Martin Burtscher (Texas State University, 2013)
**Purpose:** GPU-accelerated genetic algorithm for finding optimal finite-state machines for binary sequence prediction

---

## What This Benchmark Does

The FSM benchmark uses a **genetic algorithm (GA)** to evolve finite-state machines (FSMs) that can predict binary sequences. It's essentially teaching machines to recognize patterns in binary data through evolutionary optimization.

### The Problem

Given a sequence of binary data (0s and 1s), can we find a finite-state machine that accurately predicts the next bit? This is useful for:
- **Branch prediction** in CPUs (predicting if/else outcomes)
- **Data compression** (predicting likely values)
- **Pattern recognition** in binary streams
- **Memory access prediction**

### The Solution

Instead of manually designing the FSM, the algorithm **evolves** it using genetic programming:
1. Start with a random population of FSMs
2. Evaluate how well each FSM predicts the binary sequence
3. Keep the best FSMs and create variations (mutations and crossovers)
4. Repeat until convergence

---

## Algorithm Structure

### 1. Finite State Machine (FSM)

Each FSM has:
- **8 states** (FSMSIZE = 8)
- **2 transitions per state** (for input bit 0 or 1)
- Each transition leads to a next state (0-7)

Example FSM representation:
```
State 0: if bit=0 → go to state 3, if bit=1 → go to state 5
State 1: if bit=0 → go to state 2, if bit=1 → go to state 7
...
```

The FSM is stored as an array of 16 bytes (8 states × 2 transitions).

### 2. Prediction Process

For each bit in the input sequence:
1. Extract program counter: `pc = (data >> 1) & (TABSIZE - 1)`
2. Extract actual bit: `bit = data & 1`
3. Look up current FSM state for this PC: `state[pc]`
4. Predict bit based on current state: `prediction = state & 1`
5. Compare prediction with actual bit: count misses
6. Update state: `state[pc] = fsm[current_state * 2 + bit]`

The FSM maintains **32,768 state entries** (TABSIZE), one for each possible program counter.

### 3. Genetic Algorithm

The GA runs in parallel with:
- **1,024 populations** (POPCNT = 1024 blocks)
- **256 FSMs per population** (POPSIZE = 256 threads per block)
- **Total: 262,144 FSMs evolving in parallel**

#### Per-Population Evolution:

**Step 1: Initialization**
- Each thread creates a random FSM
- Store all 256 FSMs in shared memory (`next` array)

**Step 2: Evaluation**
- Each thread evaluates its FSM on the entire input sequence
- Count how many bits it predicts correctly
- Track misses: `misses = correct_predictions - total_bits`

**Step 3: Selection**
- Find the best FSM in the population (highest score)
- Use atomic operations to determine winner
- Track generations without improvement

**Step 4: Breeding**
- Threads with best FSM: Keep it for next generation
- 1/8 of other threads: **Mutate** best FSM (flip random bits)
- Remaining 7/8 threads: **Crossover** with best FSM (mix bits)

**Step 5: Convergence**
- Repeat until no improvement for CUTOFF generations (1 generation)
- Uses do-while loop: `while (same[blockIdx.x] < CUTOFF)`

**Step 6: Global Best**
- Compare best FSM from each population
- Keep the globally best FSM across all 1,024 populations

---

## Key Technical Characteristics

### Memory Usage

| Array | Size per Thread | Total Size | Location |
|-------|----------------|------------|----------|
| `state` | 32 KB | 8 GB | Local (stack/register) |
| `fsm` | 16 bytes | 4 MB | Local |
| `next` | 16 bytes × 256 | 4 KB per block | Shared (team-local) |
| `rndstate` | 4 bytes | 1 MB | Global |

**Critical:** Each thread needs 32 KB for `state[TABSIZE]` array!

### Synchronization Patterns

**CUDA Version Uses:**
- `__syncthreads()` - 5 barriers per generation
  - After initialization
  - After finding max score
  - After finding best thread
  - After checking improvement
  - After breeding
- `atomicMax()` - For finding best score in population
- `atomicCAS()` - For updating global best
- `__shared__` memory - For team-local FSM population

**Why This Is Complex:**
1. **Team-local memory** - `next` array shared within each block
2. **Multiple synchronization points** - 5 barriers per iteration
3. **Atomic operations** - For both max-finding and CAS updates
4. **Convergence loop** - Variable iterations (do-while with shared condition)
5. **Nested parallelism** - Blocks (teams) contain threads (parallel)

---

## Performance Characteristics

### Typical Execution (CUDA on GB10)

```
Input: 10,000 binary values
Blocks: 1,024
Threads per block: 256
Total FSMs: 262,144

Results:
- Runtime: 0.35 seconds
- Throughput: 14.87 billion transitions/second
- Baseline accuracy: 49.65% (saturating counter)
- GA accuracy: 49.74% (evolved FSM)
- Status: PASS
```

### Computational Intensity

Per generation, each thread:
1. Resets 32,768 state entries (128 KB writes)
2. Processes input sequence (e.g., 10,000 transitions)
3. Participates in 5 synchronization points
4. Performs breeding operations (16 byte operations)

With typical convergence after ~50 generations:
- **~13 billion state updates** per kernel execution
- **Highly memory-intensive** (32 KB per thread)
- **Synchronization-intensive** (5 barriers × 50 generations)

---

## Why OpenMP Port Is Difficult

### OpenMP Challenges

1. **No `__shared__` equivalent**
   - Team-local arrays exist but with limited barrier support

2. **No atomic max**
   - Must simulate with atomic read + compare + write (race conditions)

3. **No compare-and-swap**
   - Cannot implement lock-free best tracking reliably

4. **Nested parallel + barriers**
   - `#pragma omp parallel` inside `#pragma omp target teams`
   - Barriers cause compilation hangs or runtime crashes

5. **Complex control flow**
   - Do-while loop with shared condition
   - Variable iteration count based on convergence
   - Multiple synchronization points per iteration

### CUDA Advantages

| Feature | CUDA | OpenMP |
|---------|------|--------|
| Shared memory | `__shared__` | Team arrays (limited) |
| Barriers | `__syncthreads()` | `#pragma omp barrier` (buggy) |
| Atomic max | `atomicMax()` | Must simulate |
| Compare-and-swap | `atomicCAS()` | Must simulate |
| Nested parallelism | Works reliably | Crashes or fails |

---

## Comparison with Baseline

The benchmark compares against a **saturating up/down counter**, a simple 2-bit predictor:
- State 0 (strongly not taken): predicts 0
- State 1 (weakly not taken): predicts 0
- State 2 (weakly taken): predicts 1
- State 3 (strongly taken): predicts 1

The evolved GA FSM typically achieves **similar or slightly better** accuracy than this simple predictor, demonstrating that the genetic algorithm can discover effective prediction strategies.

---

## Practical Applications

### 1. Branch Prediction Research
Understanding how evolved FSMs compare to hardware predictors

### 2. Genetic Algorithm Benchmarking
Evaluating GA performance on GPUs with:
- Large populations
- Complex fitness functions
- Memory-intensive evaluation

### 3. Parallel Algorithm Study
Demonstrating:
- Massive parallelism (262K agents)
- Team-local memory patterns
- Synchronization-heavy workloads
- Convergence-based iteration

---

## Summary

**FSM_GA** is a sophisticated benchmark that:
- ✅ Tests genetic algorithm performance on GPUs
- ✅ Stresses memory subsystem (32 KB per thread)
- ✅ Requires complex synchronization (team-local barriers)
- ✅ Demonstrates massive parallelism (262K FSMs)
- ✅ Works perfectly in CUDA
- ❌ Cannot be ported to OpenMP due to compiler limitations

It represents the **upper limit** of what can be expressed in OpenMP GPU offloading, requiring features (nested parallelism with barriers, atomic max/CAS) that are either unsupported or unreliable in current implementations.
