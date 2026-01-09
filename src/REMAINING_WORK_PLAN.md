# Remaining Work - Priority Plan
**Date:** January 9, 2026
**Current Status:** 257/326 (79%)

## Remaining Issues Breakdown

| Category | Count | Difficulty | Potential Gain |
|----------|-------|------------|----------------|
| 📁 Missing data files | 38 | Easy-Medium | +10-20 |
| ❌ Runtime crashes | 23 | Hard | +5-10 |
| 🔧 Compile issues | 9 | Mixed | +3-6 |
| **Total Remaining** | **70** | | **+18-36** |

**Target:** 275-283/326 (84-87%)

---

## Priority 1: Quick Wins (Easiest Fixes)

### A. Fix Data File Paths (Est: 5-10 benchmarks, 30 min)

These have data generated but wrong paths or simple issues:

1. **b+tree-omp** - Data exists, fopen fails (check binary vs text mode)
2. **face-omp** - Data copied, but segfaults (may need different image format)
3. **multimaterial-omp** - Data generated, needs integer header added
4. **bfs-omp** - Graph generated, check format
5. **kmeans-omp** - Data generated, verify format

**Approach:**
- Check source code for exact format requirements
- Verify file permissions and paths
- Test with minimal inputs first

### B. Generate Missing Easy Data (Est: 3-5 benchmarks, 30 min)

Simple data files not yet generated:

1. **minimap2-omp** - DNA sequences (text file)
2. **svd3x3-omp** - 1M 3x3 matrices (takes time but straightforward)
3. **cmp-omp** - Seismic data (binary format)
4. **hogbom-omp** - Radio astronomy images (binary)

**Approach:**
- Generate synthetic data matching expected formats
- Start with smallest/simplest first

### C. Test Compiled Benchmarks That May Work (Est: 2-3 benchmarks, 15 min)

These compiled but we haven't tested thoroughly:

1. **grep-omp** (compiled as 'nfa') - May work with different args
2. **hybridsort-omp** - May work
3. **srad-omp** - May work

**Approach:**
- Test with various argument combinations
- Check if they complete without crashes

**Expected Gain: +10-18 benchmarks = 267-275/326 (82-84%)**

---

## Priority 2: Medium Difficulty (Moderate Effort)

### D. Debug Simple Runtime Crashes (Est: 3-5 benchmarks, 1-2 hours)

Use gdb to identify and fix segfaults:

1. **ans-omp** - Core dump (gdb analysis)
2. **qtclustering-omp** - Core dump (gdb analysis)
3. **linearprobing-omp** - Known issue (0.1% duplicate rate, may already work)

**Approach:**
```bash
gdb --args ans-omp/bin/main 10
gdb --args qtclustering-omp/qtc --Verbose
```

### E. Fix Remaining Data Formats (Est: 3-5 benchmarks, 1 hour)

Study source code and generate correct formats:

1. **cfd-omp** - CFD domain format (complex binary)
2. **diamond-omp** - FASTQ sequences (well-documented format)
3. **dxtc2-omp** - DDS texture files (DirectDraw Surface)

**Expected Gain: +6-10 benchmarks = 273-285/326 (84-87%)**

---

## Priority 3: Harder Issues (Skip or Later)

### F. Complex Data Files (Skip - Not Worth Time)

1. **leukocyte-omp** - Video file (AVI format)
2. **minibude-omp** - Molecular docking format
3. **cfd-omp** - Complex CFD format

### G. Deep Runtime Debugging (Skip - Too Time Consuming)

The 20 remaining runtime crashes likely need:
- Extensive gdb debugging
- Potential OpenMP limitations
- May not be fixable

### H. Unfixable Compile Issues (Skip)

1. **gc-omp** - OpenMP translation error (fundamental)
2. **miniFE-omp** - No Makefile
3. **miniWeather-omp** - Needs MPI

---

## Recommended Action Plan

### Session Goals (2-3 hours)

**Phase 1: Quick Wins (30-60 min)**
1. Fix b+tree-omp data file issue
2. Fix multimaterial-omp format (add integer header)
3. Test grep/hybridsort/srad compiled benchmarks
4. Generate minimap2 data

**Phase 2: Medium Wins (30-60 min)**
5. Debug ans-omp with gdb
6. Fix face-omp segfault
7. Generate remaining easy data files
8. Test linearprobing-omp (may already work)

**Phase 3: Stretch Goals (30-60 min)**
9. Generate svd3x3 data (1M matrices, takes time)
10. Generate diamond-omp FASTQ data
11. Debug 1-2 more crashes with gdb

**Expected Final Result: 275-283/326 (84-87%)**

---

## Effort vs Reward Analysis

| Priority | Effort | Benchmarks | Success Rate | ROI |
|----------|--------|------------|--------------|-----|
| **Quick Wins** | 1 hour | +10-15 | 82-84% | High |
| **Medium Wins** | 2 hours | +6-10 | 84-87% | Medium |
| **Hard Issues** | 5+ hours | +5-10 | 87-90% | Low |

**Recommendation:** Focus on Priority 1 & 2, skip Priority 3.

---

## Detailed Next Steps

### 1. b+tree-omp (Immediate)
```bash
# Check if command.txt should be binary
grep "fopen.*command_file.*rb" b+tree-omp/main.c
# Try regenerating as binary if needed
```

### 2. multimaterial-omp (Immediate)
```python
# Add integer header to volfrac.dat
import numpy as np
nmats = 10
with open('multimaterial-omp/volfrac.dat', 'rb+') as f:
    # Read existing data
    data = np.fromfile(f, dtype=np.float64)
    # Rewrite with header
    f.seek(0)
    np.array([nmats], dtype=np.int32).tofile(f)
    data.tofile(f)
```

### 3. Test Compiled Benchmarks
```bash
# grep-omp
./grep-omp/nfa pattern testfile

# hybridsort-omp
./hybridsort-omp/hybridsort

# srad-omp
./srad-omp/srad 1000 1000 0 127 0 127 0.5 2
```

### 4. Generate minimap2 Data
```python
# Simple DNA sequences
with open('../minimap2-sycl/in-1k.txt', 'w') as f:
    for i in range(1000):
        seq = ''.join(random.choice('ACGT') for _ in range(100))
        f.write(f">seq{i}\n{seq}\n")
```

---

## Success Metrics

**Minimum Success:** 275/326 (84.4%) - +18 benchmarks
**Target Success:** 280/326 (85.9%) - +23 benchmarks
**Stretch Success:** 285/326 (87.4%) - +28 benchmarks

Current: 257/326 (78.8%)

**Let's aim for 84-86% (275-283 benchmarks working)**
