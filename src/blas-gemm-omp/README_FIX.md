# blas-gemm-omp Fix

## Original Issue
The benchmark originally required Intel MKL (Math Kernel Library) with OpenMP offload extensions (`mkl.h` and `mkl_omp_offload.h`), which are not available when using NVIDIA HPC SDK.

## Solution
Replaced Intel MKL calls with NVIDIA cuBLAS library, which is available in the NVIDIA HPC SDK.

## Changes Made

### 1. Replaced Headers
- **Removed:** `#include "mkl.h"` and `#include "mkl_omp_offload.h"`
- **Added:** `#include <cublas_v2.h>`

### 2. Replaced Memory Allocation
- **Original:** `mkl_malloc()` / `mkl_free()`
- **New:** Standard `malloc()` / `free()`

### 3. Replaced BLAS Calls
- **Original:** Used `#pragma omp dispatch` with `sgemm()` / `dgemm()` / `hgemm()`
- **New:** Direct cuBLAS calls:
  - `cublasSgemm()` for single precision
  - `cublasDgemm()` for double precision
  - `cublasHgemm()` for half precision

### 4. Data Management
- Use OpenMP `target enter/exit data` and `target update` for device memory management
- Use `use_device_addr` to get device pointers for cuBLAS calls

### 5. Updated Makefile
- Added cuBLAS include path: `-I/opt/nvidia/hpc_sdk/.../math_libs/.../include`
- Added cuBLAS library: `-cudalib=cublas`

## Compatibility
- **Original:** Intel OneAPI with Intel GPUs
- **New:** NVIDIA HPC SDK with NVIDIA GPUs (GB10/sm_121 tested)

## Performance
Both single and double precision GEMM operations work correctly with cuBLAS, providing optimized performance on NVIDIA GPUs.

## Files Modified
- `main.cpp` - Completely rewritten to use cuBLAS instead of MKL
- `Makefile` - Added cuBLAS include path and linker flags
- `main.cpp.mkl_original` - Backup of original MKL version

## Testing
```bash
make clean && make
./main 79 91 83 10       # Small test
./main 4096 4096 4096 100  # Large test
```

Both tests pass successfully with correct output.
