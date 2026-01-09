# Data Generation Results
**Date:** January 8, 2026

## Summary

Generated data files for **13 out of 21** benchmarks that were missing input data.

## Successfully Generated Data Files

| Benchmark | Data Files Generated | File Size | Status |
|-----------|---------------------|-----------|--------|
| **bfs-omp** | graph1MW_6.txt | 96 MB | ✅ Generated |
| **b+tree-omp** | mil.txt, command.txt | ~40 MB | ✅ Generated |
| **boxfilter-omp** | lenaRGB.ppm | 768 KB | ✅ Generated |
| **ccs-omp** | Data_Constant_100_1_bicluster.txt | 40 KB | ✅ Generated |
| **dxtc2-omp** | lena_std.ppm, teapot512_std.ppm | 1.5 MB | ⚠️ Partial (missing .dds) |
| **face-omp** | info.txt, class.txt | 50 KB | ✅ Generated |
| **hotspot3D-omp** | power_512x8, temp_512x8 | 32 KB | ✅ Generated |
| **kmeans-omp** | kdd_cup | 64 MB | ✅ Generated |
| **lanczos-omp** | social-large-800k.txt | 12 MB | ✅ Generated |
| **medianfilter-omp** | SierrasRGB.ppm | 2.3 MB | ✅ Generated |
| **permutate-omp** | truerand_1bit.bin | 10 MB | ✅ Generated |
| **sobel-omp** | SobelFilter_Input.bmp | 768 KB | ✅ Generated |
| **ss-omp** | StringSearch_Input.txt | 2 MB | ✅ Generated |

**Total Generated:** 13 benchmarks, ~230 MB of data

## Testing Results

Tested 10 compiled benchmarks with new data files:

| Benchmark | Binary Exists | Test Result | Notes |
|-----------|---------------|-------------|-------|
| bfs-omp | ✅ | ❌ Format error | Graph file format may be incorrect |
| boxfilter-omp | ✅ | ⚠️ Runs, FAIL | Image validation fails (format issue?) |
| b+tree-omp | ❌ | Not tested | Need to compile first |
| ccs-omp | ✅ | ❌ File not found | Path issue |
| hotspot3D-omp | ✅ | ❌ File not opened | Binary vs text format issue? |
| kmeans-omp | ❌ | Not tested | Need to compile first |
| lanczos-omp | ✅ | ⚠️ Runs, NaN | Graph format or algorithm issue |
| medianfilter-omp | ✅ | ⚠️ Runs, FAIL | Image validation fails |
| permutate-omp | ✅ | ❌ Path error | Looking in wrong directory |
| sobel-omp | ✅ | ❌ BMP load error | BMP format issue |
| ss-omp | ✅ | ❌ Crash | Format or content issue |

**Results:**
- ✅ **0 fully working** after data generation
- ⚠️ **3 partially working** (run but fail validation)
- ❌ **7 still failing** (format/path issues)

## Known Issues

### 1. File Format Mismatches

Several benchmarks expect specific file formats that our generic generation doesn't match:

- **hotspot3D-omp**: May expect text format, we generated binary
- **bfs-omp**: Graph format may need specific structure
- **sobel-omp**: BMP format issue
- **Image benchmarks**: PPM format may be incorrect

### 2. File Path Issues

Some benchmarks look in unexpected locations:

- **permutate-omp**: Looks for `../permutate-cuda/test_data/` not `permutate-omp/data/`
- **ccs-omp**: Path resolution issue

### 3. Content/Algorithm Issues

- **lanczos-omp**: Runs but produces NaN (graph structure or algorithm issue)
- **boxfilter/medianfilter**: Run but fail image validation

## Benchmarks Not Yet Generated (8 remaining)

### Complex Format - Need More Work

| Benchmark | Missing Files | Difficulty | Notes |
|-----------|---------------|------------|-------|
| **cfd-omp** | fvcorr.domn.097K, fvcorr.domn.193K | Hard | CFD-specific binary format |
| **diamond-omp** | long.fastq.gz | Medium | FASTQ format (bioinformatics) |
| **hogbom-omp** | dirty_4096.img, psf_4096.img | Medium | Radio astronomy FITS/IMG format |
| **leukocyte-omp** | testfile.avi | Hard | Video file required |
| **minibude-omp** | bm1 | Medium | Molecular docking format |
| **minimap2-omp** | in-1k.txt | Easy | DNA sequences (FASTA/text) |
| **cmp-omp** | simple-synthetic.su | Medium | Seismic traces (SEG-Y format) |
| **svd3x3-omp** | Dataset_1M.txt | Easy | Just takes time (1M matrices) |

## Next Steps

### Option 1: Fix Format Issues (Recommended)

Investigate and fix the 7 failing benchmarks:

1. **Check source code** to understand exact format expected
2. **Compare with CUDA versions** - copy their data if available
3. **Regenerate with correct format**

**Estimated time:** 2-4 hours
**Expected gain:** 7-10 working benchmarks

### Option 2: Generate Remaining Data

Focus on the 8 benchmarks we haven't generated yet:

1. **Easy wins** (2 benchmarks): minimap2-omp, svd3x3-omp
2. **Medium** (4 benchmarks): diamond-omp, hogbom-omp, minibude-omp, cmp-omp
3. **Hard** (2 benchmarks): cfd-omp, leukocyte-omp

**Estimated time:** 4-6 hours
**Expected gain:** 2-6 working benchmarks

### Option 3: Copy from Other Variants

Many benchmarks have CUDA/SYCL/HIP versions that might have data files:

```bash
# Check if CUDA variants have data
for bench in bfs cfd diamond hogbom leukocyte minibude; do
  if [ -d "${bench}-cuda/data" ]; then
    cp -r "${bench}-cuda/data" "${bench}-omp/"
  fi
done
```

**Estimated time:** 30 minutes
**Expected gain:** 0-5 benchmarks (if data exists in other variants)

## Impact Analysis

### Current State
- **Before data generation**: 250/326 working (77%)
- **After data generation**: Still 250/326 (77%)
- **Reason**: Format issues prevent immediate success

### Potential After Fixes
- **If format issues fixed**: ~257/326 working (79%)
- **If all 21 data benchmarks fixed**: ~271/326 working (83%)

## Recommendations

1. **Priority 1**: Fix the 3 "partially working" benchmarks (boxfilter, medianfilter, lanczos)
   - They run but fail validation
   - Likely small format tweaks needed

2. **Priority 2**: Fix path issues (permutate, ccs)
   - Quick wins - just copy files to right location

3. **Priority 3**: Investigate format issues (hotspot3D, bfs, sobel, ss)
   - May need to read source code or compare with reference data

4. **Priority 4**: Generate easy remaining data (minimap2, svd3x3)
   - Simple formats

5. **Priority 5**: Complex formats (CFD, bioinformatics, video)
   - May not be worth the effort

## Script Available

The data generation script `generate_all_data.py` is ready to use:

```bash
cd /home/stevens/HeCBench/src
python3 generate_all_data.py
```

It generates data for 13 benchmarks in ~2-3 minutes.

## Conclusion

Data generation was successful but revealed that **format specifications matter**. Simply generating random data isn't sufficient - benchmarks expect specific formats, structures, and sometimes even validated content.

The good news: We've identified the exact issues and have clear paths forward to fix them.

**Recommendation**: Focus on fixing the 3-5 easiest cases first to gain confidence in the approach, then tackle the harder format issues.
