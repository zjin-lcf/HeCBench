# HeCBench Renovation Plan

**Date**: 2025-12-06
**Purpose**: Comprehensive roadmap for modernizing and enhancing the HeCBench heterogeneous computing benchmark suite

---

## Executive Summary

HeCBench is a mature benchmark suite with **508 benchmarks** across **26 categories**, supporting 4 primary programming models (CUDA, HIP, SYCL, OpenMP). The suite demonstrates excellent breadth and multi-model coverage (95%+ for GPU models) but requires modernization in build infrastructure, tooling, and systematic performance analysis capabilities.

**Key Metrics**:
- 1,820 benchmark implementations across 508 unique benchmarks
- 2,587 independent Makefiles (current build system)
- 97.8% CUDA coverage, 96.9% HIP, 94.5% SYCL, 63.4% OpenMP
- Active maintenance with recent ROCm7 and CUDA 13 compatibility updates

---

## Phase 1: Review and Recommendations

### Current State Assessment

#### Strengths
1. **Comprehensive coverage**: 26 categories spanning simulation, ML, math, computer vision, cryptography, etc.
2. **Strong multi-model support**: Near-complete CUDA/HIP/SYCL implementations
3. **Consistent code quality**: Uniform licensing, style patterns, and organization
4. **Active maintenance**: Recent compatibility fixes for ROCm7, CUDA 13
5. **Automation infrastructure**: Python scripts for bulk testing (`autohecbench.py`)
6. **Well-documented**: 75KB README with installation, usage, and benchmark references

#### Critical Weaknesses
1. **Build system fragmentation**: 2,587 independent Makefiles with no top-level orchestration
2. **No unified configuration**: Each benchmark requires manual compiler/architecture setup
3. **Incomplete OpenMP coverage**: Only 63% of benchmarks have OpenMP implementations
4. **Limited testing infrastructure**: No automated test suite or CI/CD
5. **Missing performance baselines**: No documented expected performance ranges
6. **Manual dependency management**: No package manager integration

### Detailed Findings by Area

#### 1.1 Programming Model Coverage

**Current Distribution**:
| Model | Coverage | Gaps |
|-------|----------|------|
| CUDA | 505/508 (99.4%) | 3 benchmarks missing |
| HIP | 500/508 (98.4%) | 8 benchmarks missing |
| SYCL | 486/508 (95.7%) | 22 benchmarks missing |
| OpenMP 4.5 | 327/508 (64.4%) | 181 benchmarks missing |

**Critical Gaps**:
- **Single-implementation benchmarks** (5): `dwt2d`, `intrinsics-simd`, `local-ht`, `miniDGS` (CUDA only)
- **Incomplete quad-coverage** (19 benchmarks): Missing 1-2 implementations
  - Examples: `addBiasQKV`, `bicgstab`, `blas-fp8gemm`, `btree`, `clock`
  - Sparse linear algebra under-represented: `sddmm-batch`, `spaxpby`, `spnnz`, `spmm`

**Recommendation**: Prioritize completing quad-coverage for high-value benchmarks (ML, sparse algebra)

#### 1.2 Build System Analysis

**Current Architecture**:
- **GNU Make** with per-benchmark Makefiles
- **No dependency tracking** between benchmarks
- **Manual configuration** via environment variables or Makefile edits
- **Compiler-specific variants**: `Makefile.nvc`, `Makefile.aomp` for OpenMP

**Pain Points**:
- No bulk compilation capability
- Difficult to maintain consistent compiler flags across benchmarks
- Complex architecture targeting (sm_60, sm_70, sm_80, sm_90 for CUDA)
- Rebuild entire benchmark for configuration changes

**Recommendation**: Migrate to CMake with presets for compiler/architecture combinations

#### 1.3 Testing & Validation

**Current Approach**:
- **Runtime verification**: Regex-based output parsing in `autohecbench.py`
- **Manual correctness checks**: Reference implementations in some benchmarks
- **No automated regression testing**

**Issues**:
- README notes: "Not all programs automate verification"
- No performance regression detection
- No cross-model correctness validation

**Recommendation**: Implement automated test suite with correctness + performance regression checks

#### 1.4 Documentation Quality

**Strengths**:
- Comprehensive main README.md
- 242 benchmark-specific README files
- Recent research citations (2023-2025)

**Gaps**:
- No performance tuning guides
- No expected performance baselines
- No GPU/device requirement matrix
- Missing architectural optimization notes

**Recommendation**: Add performance characterization and tuning documentation

### Phase 1 Improvement Priorities

#### Priority 1: Critical Infrastructure (Enables Phase 2+)
- ✅ Complete exploration and assessment (DONE)
- ✅ Migrate to CMake build system (98% COMPLETE - 1,790/1,818 implementations)
- ✅ Implement build presets for common configurations (DONE)
- ✅ Create unified CLI tool (`tools/hecbench`) (DONE)
- ✅ Add CTest integration for automated testing (DONE)
- ✅ Create benchmark metadata system (`benchmarks.yaml`) (DONE)
- ✅ Implement result collection framework (`tools/hecbench_results.py`) (DONE)

#### Priority 2: Coverage Expansion
- Complete quad-coverage for 19 partially-implemented benchmarks
- Port 5 single-implementation benchmarks to additional models
- Add OpenMP implementations for high-value missing benchmarks (~50 priorities)

#### Priority 3: Quality Assurance
- Create automated correctness test suite
- Implement performance baseline tracking
- Add cross-model result validation

#### Priority 4: Developer Experience
- Simplify compiler toolchain setup
- Add containerized build environments
- Document expected performance ranges

---

## Phase 2: CMake Build System Migration

### Objectives
1. Replace 2,587 Makefiles with unified CMake build system
2. Enable selective benchmark compilation (by name, category, programming model)
3. Implement configuration presets for common compiler/architecture combinations
4. Maintain backward compatibility during transition

### Proposed Architecture

#### Top-Level Structure
```
HeCBench/
├── CMakeLists.txt                 # Root build configuration
├── cmake/
│   ├── presets/
│   │   ├── cuda-sm80.cmake       # NVIDIA A100 preset
│   │   ├── hip-gfx90a.cmake      # AMD MI250X preset
│   │   ├── sycl-cuda.cmake       # SYCL with CUDA backend
│   │   └── openmp-intel.cmake    # Intel OpenMP offload
│   ├── FindHIP.cmake             # Compiler detection
│   ├── FindSYCL.cmake
│   └── BenchmarkMacros.cmake     # add_hecbench_benchmark()
├── CMakePresets.json             # User-facing presets
└── src/
    ├── CMakeLists.txt            # Benchmark discovery
    └── benchmark-{model}/
        └── CMakeLists.txt        # Per-benchmark config
```

#### Key Features

**1. Selective Building**
```bash
# Build all benchmarks for CUDA
cmake --preset cuda-sm80
cmake --build build/

# Build specific benchmark across all models
cmake --build build/ --target jacobi-all

# Build specific model variant
cmake --build build/ --target jacobi-cuda

# Build category
cmake --build build/ --target category-ml
```

**2. CMake Presets**
```json
{
  "configurePresets": [
    {
      "name": "cuda-sm80",
      "displayName": "NVIDIA A100 (sm_80)",
      "cacheVariables": {
        "HECBENCH_ENABLE_CUDA": "ON",
        "HECBENCH_CUDA_ARCH": "sm_80",
        "CMAKE_CUDA_COMPILER": "nvcc"
      }
    },
    {
      "name": "hip-gfx90a",
      "displayName": "AMD MI250X (gfx90a)",
      "cacheVariables": {
        "HECBENCH_ENABLE_HIP": "ON",
        "HECBENCH_HIP_ARCH": "gfx90a",
        "CMAKE_HIP_COMPILER": "hipcc"
      }
    },
    {
      "name": "all-models",
      "displayName": "Build all programming models",
      "cacheVariables": {
        "HECBENCH_ENABLE_CUDA": "ON",
        "HECBENCH_ENABLE_HIP": "ON",
        "HECBENCH_ENABLE_SYCL": "ON",
        "HECBENCH_ENABLE_OPENMP": "ON"
      }
    }
  ]
}
```

**3. Benchmark Registration Macro**
```cmake
# src/jacobi-cuda/CMakeLists.txt
add_hecbench_benchmark(
  NAME jacobi
  MODEL cuda
  SOURCES main.cu kernel.cu
  CATEGORIES simulation math
  ARCHITECTURES sm_60 sm_70 sm_80 sm_90
  VERIFICATION ON
  REFERENCE_OUTPUT "Residual: 1.234e-05"
)
```

**4. Dependency Management**
```cmake
# Automatic compiler detection
find_package(CUDAToolkit REQUIRED)
find_package(HIP QUIET)
find_package(SYCL QUIET)
find_package(OpenMP QUIET)

# Optional library discovery
find_package(oneDPL QUIET)
find_package(TBB QUIET)
```

### Implementation Strategy

#### Step 1: Proof of Concept (1 week)
- Implement root `CMakeLists.txt`
- Create macro library (`BenchmarkMacros.cmake`)
- Convert 10 representative benchmarks:
  - Simple: `jacobi`, `nbody`, `softmax`
  - Complex: `attention`, `miniDGS`, `halo-finder`
  - Multi-file: `lulesh`, `haccmk`
- Test selective building

#### Step 2: Category Migration (2-3 weeks)
- Convert benchmarks by category:
  - Week 1: Simulation, Math (165 benchmarks)
  - Week 2: ML, Computer Vision (120 benchmarks)
  - Week 3: Remaining categories (223 benchmarks)
- Maintain Makefile alongside CMake during transition

#### Step 3: Preset Development (1 week)
- Create presets for common GPUs:
  - NVIDIA: V100, A100, H100, H200
  - AMD: MI100, MI250X, MI300A/X
  - Intel: Data Center GPU Max
- Test cross-compiler compatibility

#### Step 4: Validation & Cutover (1 week)
- Run full test suite with both Make and CMake
- Verify binary compatibility
- Update documentation
- Archive Makefiles (optional removal)

### Expected Benefits

1. **Faster builds**: Parallel compilation across benchmarks, incremental rebuilds
2. **Simpler configuration**: Single preset selection vs. editing 100+ Makefiles
3. **Better IDE support**: CLion, VS Code CMake integration
4. **Dependency tracking**: Automatic rebuilds when shared headers change
5. **Easier CI/CD**: Standard CMake workflows for GitHub Actions, GitLab CI

### Risk Mitigation

**Risk 1**: Complex dependency chains in some benchmarks
**Mitigation**: Incremental migration with parallel Makefile support during transition

**Risk 2**: Compiler-specific flag requirements
**Mitigation**: Per-compiler flag databases in CMake modules

**Risk 3**: User resistance to change
**Mitigation**: Maintain Makefiles for 6-12 months, provide migration guide

---

## Phase 3: Tooling for Benchmark Management

### Objectives
1. Provide high-level tools for managing benchmark campaigns
2. Simplify execution across multiple GPUs, compilers, and configurations
3. Enable systematic performance analysis and visualization
4. Improve user experience for large-scale experimentation

### Proposed Tools

#### 3.1 HecBench Runner (Enhanced `autohecbench.py`)

**Current Capabilities**:
- Bulk compilation and execution
- Regex-based result extraction
- Comparison between runs

**Proposed Enhancements**:

**A. Multi-GPU Support**
```bash
# Run benchmarks on specific GPUs
hecbench run --gpus 0,1,2,3 --category ml

# Distributed execution across cluster
hecbench run --cluster slurm --nodes 4 --category simulation
```

**B. Campaign Management**
```bash
# Define experiment
hecbench campaign create \
  --name "a100-vs-mi250x" \
  --benchmarks category:ml,category:math \
  --configs cuda-sm80,hip-gfx90a \
  --iterations 10

# Execute campaign
hecbench campaign run a100-vs-mi250x

# Analyze results
hecbench campaign report a100-vs-mi250x --format html
```

**C. Result Database**
```bash
# Store results in SQLite database
hecbench run --category ml --store results.db

# Query historical results
hecbench query --benchmark attention --gpu A100 --since 2024-01-01

# Export to CSV/JSON
hecbench export results.db --format csv --output results.csv
```

#### 3.2 Performance Visualization Dashboard

**Features**:
1. **Comparative charts**: Bar charts comparing performance across models/GPUs
2. **Trend analysis**: Performance regression detection over time
3. **Heatmaps**: Category-level coverage and performance matrices
4. **Roofline plots**: Computational intensity vs. achieved performance
5. **Interactive exploration**: Filter by category, GPU, benchmark

**Implementation Options**:
- **Option A**: Web-based dashboard (React + D3.js)
- **Option B**: Python Jupyter notebooks (matplotlib, plotly)
- **Option C**: Static HTML reports (automated generation)

**Recommended**: Hybrid approach - static HTML for CI/CD, Jupyter for deep analysis

#### 3.3 Containerized Environments

**Purpose**: Simplify compiler and library installation

**Proposed Containers**:
```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y cmake ninja-build
COPY . /hecbench
WORKDIR /hecbench
RUN cmake --preset cuda-sm80 && cmake --build build/

# Dockerfile.hip
FROM rocm/dev-ubuntu-22.04:6.0
RUN apt-get update && apt-get install -y cmake ninja-build
COPY . /hecbench
WORKDIR /hecbench
RUN cmake --preset hip-gfx90a && cmake --build build/

# Dockerfile.sycl
FROM intel/oneapi-hpckit:2024.1
COPY . /hecbench
WORKDIR /hecbench
RUN cmake --preset sycl-cuda && cmake --build build/
```

**Usage**:
```bash
# Build in container
docker run --gpus all -v $PWD:/workspace hecbench:cuda cmake --build build/

# Run benchmarks in container
docker run --gpus all hecbench:cuda hecbench run --category ml
```

#### 3.4 Continuous Integration Dashboard

**Features**:
1. **Automated testing**: Run benchmarks on every commit
2. **Performance tracking**: Detect regressions across GPUs
3. **Compatibility monitoring**: Test multiple compiler versions
4. **Coverage reports**: Track implementation completeness

**GitHub Actions Workflow**:
```yaml
name: HeCBench CI

on: [push, pull_request]

jobs:
  test-cuda:
    runs-on: [self-hosted, gpu, a100]
    steps:
      - uses: actions/checkout@v3
      - name: Build CUDA benchmarks
        run: cmake --preset cuda-sm80 && cmake --build build/
      - name: Run tests
        run: ctest --test-dir build/ --output-on-failure
      - name: Check performance
        run: hecbench perf-check --baseline results/baseline-a100.json

  test-hip:
    runs-on: [self-hosted, gpu, mi250x]
    steps:
      - uses: actions/checkout@v3
      - name: Build HIP benchmarks
        run: cmake --preset hip-gfx90a && cmake --build build/
      - name: Run tests
        run: ctest --test-dir build/ --output-on-failure
```

#### 3.5 Documentation Generator

**Automatic Generation**:
- Benchmark catalog with descriptions, categories, implementations
- Performance baselines per GPU
- Tuning guides from code annotations

**Implementation**:
```bash
# Extract metadata from benchmarks
hecbench docs generate --output docs/

# Generates:
# - docs/index.html (catalog)
# - docs/benchmarks/{name}.html (per-benchmark pages)
# - docs/categories/{category}.html (category summaries)
# - docs/performance-baselines.html (expected ranges)
```

### Implementation Roadmap

**Week 1-2**: Enhanced Runner
- Multi-GPU support
- Campaign management
- Result database (SQLite)

**Week 3-4**: Visualization
- Python plotting library integration
- Static HTML report generation
- Jupyter notebook templates

**Week 5-6**: Containers & CI
- Dockerfile creation for CUDA, HIP, SYCL
- GitHub Actions workflows
- Performance regression checks

**Week 7-8**: Documentation & Polish
- Auto-generate benchmark catalog
- User guides for new tools
- Integration testing

---

## Phase 4: Serial CPU Implementation & Coverage Expansion

### 4.1 Serial CPU Implementation Strategy

**Objective**: Create reference serial CPU implementations for all 508 benchmarks

**Rationale**:
1. **Performance baselines**: Measure GPU speedup vs. single-threaded CPU
2. **Correctness validation**: Verify GPU results against simple CPU reference
3. **Accessibility**: Enable testing without GPU hardware
4. **Educational**: Understand algorithm before GPU optimization

**Implementation Approach**:

**Option A: Derive from OpenMP**
```cpp
// Convert OpenMP to serial by removing pragmas
// Before (jacobi-omp/main.cpp):
#pragma omp target teams distribute parallel for
for (int i = 1; i < n-1; i++) {
  u_new[i] = 0.5 * (u[i-1] + u[i+1]);
}

// After (jacobi-serial/main.cpp):
for (int i = 1; i < n-1; i++) {
  u_new[i] = 0.5 * (u[i-1] + u[i+1]);
}
```

**Pros**: Fast conversion (327 benchmarks already have OpenMP)
**Cons**: May miss optimizations in non-OpenMP benchmarks

**Option B: Simplify GPU kernels**
```cpp
// Translate CUDA kernel to serial C++
// Before (jacobi-cuda/kernel.cu):
__global__ void jacobi_kernel(float* u, float* u_new, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i < n-1) {
    u_new[i] = 0.5 * (u[i-1] + u[i+1]);
  }
}

// After (jacobi-serial/main.cpp):
void jacobi_serial(float* u, float* u_new, int n) {
  for (int i = 1; i < n-1; i++) {
    u_new[i] = 0.5 * (u[i-1] + u[i+1]);
  }
}
```

**Pros**: Works for all benchmarks, educational value
**Cons**: More manual effort (505 CUDA benchmarks)

**Recommended Hybrid Approach**:
1. Derive from OpenMP where available (327 benchmarks) - **automated**
2. Simplify CUDA for remaining 181 benchmarks - **semi-automated with script**
3. Manual review for complex cases (miniDGS, halo-finder, etc.)

**Automation Script**:
```python
# tools/generate_serial.py
def convert_omp_to_serial(benchmark_path):
    """Remove OpenMP pragmas and offload directives"""
    source = read_file(f"{benchmark_path}/main.cpp")

    # Remove pragmas
    source = re.sub(r'#pragma omp.*\n', '', source)

    # Update target type if needed
    source = source.replace('omp target', '')

    # Write serial version
    write_file(f"{benchmark_path.replace('-omp', '-serial')}/main.cpp", source)

def convert_cuda_to_serial(benchmark_path):
    """Simplify CUDA kernel to CPU loop"""
    # Parse kernel
    kernel = extract_cuda_kernel(f"{benchmark_path}/kernel.cu")

    # Convert to serial loop
    serial_code = cuda_to_serial_template(kernel)

    # Generate complete program
    write_serial_benchmark(benchmark_path, serial_code)
```

**Validation**:
- Run serial implementation
- Compare results with GPU implementations
- Verify numerical accuracy (tolerance for floating-point differences)

### 4.2 New Benchmark Recommendations

**Coverage Gap Analysis**:

Based on review of current 26 categories and 508 benchmarks, the following areas are under-represented:

#### Gap 1: Modern AI/ML Workloads

**Missing Benchmarks**:
1. **Transformer Components**:
   - Multi-head attention with flash attention optimization
   - RoPE (Rotary Position Embedding)
   - Grouped-query attention (GQA)
   - Layer normalization variants (RMSNorm)

2. **Quantization Operations**:
   - INT8/INT4 quantization/dequantization
   - Mixed-precision GEMM (FP16/BF16/FP8)
   - Quantized attention

3. **ML Optimizers**:
   - Adam/AdamW
   - LAMB optimizer
   - Distributed gradient allreduce patterns

**Recommended Additions** (10 benchmarks):
- `flash-attention` - Flash Attention v2 implementation
- `rope` - Rotary position embedding
- `gqa` - Grouped-query attention
- `rmsnorm` - RMS normalization
- `quantize-int8` - INT8 quantization
- `quantize-int4` - INT4 quantization with grouping
- `adam-optimizer` - Adam optimizer step
- `lamb-optimizer` - LAMB optimizer
- `allreduce` - Distributed gradient reduction
- `mixed-precision-gemm` - FP8/FP16/BF16 GEMM variants

#### Gap 2: Sparse and Irregular Computation

**Current Coverage**: Limited (4 sparse benchmarks: `sddmm-batch`, `spaxpby`, `spnnz`, `spmm`)

**Missing Patterns**:
1. **Sparse formats**: COO, CSR, CSC, ELL, HYB conversions
2. **Graph analytics**: PageRank, community detection, triangle counting
3. **Sparse neural networks**: Pruning, structured sparsity

**Recommended Additions** (8 benchmarks):
- `sparse-format-convert` - COO/CSR/CSC/ELL conversions
- `pagerank` - PageRank algorithm
- `triangle-count` - Triangle counting for graphs
- `community-detection` - Louvain algorithm
- `sparse-attention` - Sparse attention patterns (block-sparse)
- `pruning` - Neural network pruning
- `structured-sparsity` - 2:4 structured sparsity patterns
- `sparse-mlp` - Sparse MLP forward/backward

#### Gap 3: Data Movement and Communication

**Current Coverage**: Minimal (focused on compute)

**Missing Patterns**:
1. **Inter-GPU communication**: P2P transfers, NVLink/Infinity Fabric patterns
2. **Host-device transfers**: Pinned memory, async transfers
3. **Compression**: On-the-fly compression for data transfer

**Recommended Additions** (6 benchmarks):
- `p2p-transfer` - Peer-to-peer GPU transfers
- `nvlink-bandwidth` - NVLink bandwidth microbenchmark
- `async-memcpy` - Overlapped compute and transfer
- `pinned-memory` - Pinned vs. pageable memory comparison
- `compression-transfer` - Compressed data transfer (LZ4, Snappy)
- `uvm-patterns` - Unified virtual memory access patterns

#### Gap 4: Emerging Hardware Features

**Missing Coverage**:
1. **Tensor Core operations**: Beyond WMMA (currently 1 benchmark)
2. **Ray tracing acceleration**: RT cores
3. **Hardware-accelerated encryption**: AES-NI beyond basic AES

**Recommended Additions** (6 benchmarks):
- `tensor-core-fp8` - FP8 Tensor Core operations
- `tensor-core-int8` - INT8 Tensor Core operations
- `mma-shapes` - Various MMA shapes (m16n8k16, etc.)
- `rt-bvh-build` - BVH construction for ray tracing
- `rt-traversal` - Ray-BVH traversal
- `crypto-accelerated` - Hardware-accelerated cryptographic operations

#### Gap 5: Scientific Computing Patterns

**Current Coverage**: Good but missing specific methods

**Missing Algorithms**:
1. **Advanced linear solvers**: Multigrid, preconditioners
2. **FFT variants**: 2D/3D FFT (only 1D currently)
3. **PDE solvers**: Specific methods (CG, BiCGSTAB have limited coverage)

**Recommended Additions** (5 benchmarks):
- `multigrid` - Multigrid solver
- `fft-2d` - 2D FFT
- `fft-3d` - 3D FFT
- `preconditioner-jacobi` - Jacobi preconditioner
- `preconditioner-ilu` - Incomplete LU preconditioner

### Prioritized Benchmark Addition Plan

**High Priority** (20 benchmarks - critical for modern GPU computing):
1. Flash Attention (AI/ML)
2. Quantization (INT8, INT4)
3. Sparse format conversions
4. P2P GPU transfers
5. Tensor Core FP8/INT8
6. 2D/3D FFT

**Medium Priority** (15 benchmarks - fill important gaps):
7. Graph analytics (PageRank, triangle counting)
8. Communication patterns (async memcpy, compression)
9. ML optimizers (Adam, LAMB)
10. Sparse neural networks

**Low Priority** (10 benchmarks - nice-to-have):
11. Ray tracing acceleration
12. Advanced preconditioners
13. Structured sparsity patterns

**Total New Benchmarks**: 35-45 (7-9% expansion of suite)

---

## Phase 5: Future-Proofing for Extreme Heterogeneity

### 5.1 Architecture Trends (2025-2030)

**Predicted Landscape**:

1. **Accelerator Diversity**:
   - NVIDIA: Hopper → Blackwell → next-gen
   - AMD: CDNA3 → CDNA4/RDNA4 hybrid → unified architecture
   - Intel: Ponte Vecchio → Falcon Shores (GPU+CPU tiles)
   - Custom ASICs: Google TPU v6+, AWS Trainium2/Inferentia3, Cerebras WSE-3

2. **Heterogeneous Compute Tiles**:
   - CPU + GPU + NPU (neural processing unit) on-package
   - Specialized matrix engines (systolic arrays, spatial architectures)
   - Optical interconnects for chiplet communication

3. **Programming Model Evolution**:
   - SYCL 2025+ with more backend targets
   - Unified memory everywhere (CXL, HBM)
   - Compiler-managed data movement
   - Domain-specific languages (DSLs) compiling to heterogeneous targets

4. **Workload Characteristics**:
   - Extreme sparsity (99%+ zeros in AI models)
   - Mixed precision (FP4, block floating point)
   - Hybrid data types within single operation
   - Dynamic execution graphs

### 5.2 Future-Proofing Strategies

#### Strategy 1: Abstract Programming Model Interface

**Objective**: Decouple benchmarks from specific programming models

**Approach**: Define HecBench Abstraction Layer (HBAL)

```cpp
// hbal.h - Unified API for heterogeneous computing
namespace hbal {
  // Device management
  Device get_device(int id);
  void set_device(Device dev);

  // Memory management
  template<typename T>
  DeviceBuffer<T> allocate(size_t n, Device dev);

  template<typename T>
  void copy(T* dst, const T* src, size_t n, CopyKind kind);

  // Kernel execution
  template<typename Kernel, typename... Args>
  void launch(Kernel k, GridConfig grid, Args... args);

  // Synchronization
  void synchronize(Device dev);
  Event record_event(Device dev);
  void wait_event(Event e);
}

// Backend implementations:
// - hbal_cuda.cpp (NVIDIA GPUs)
// - hbal_hip.cpp (AMD GPUs)
// - hbal_sycl.cpp (Intel GPUs, CPUs)
// - hbal_openmp.cpp (CPUs with offload)
// - hbal_metal.cpp (Apple Silicon)
// - hbal_npu.cpp (Neural processing units)
```

**Benefits**:
1. Single benchmark source → multiple backends
2. Easy to add new hardware targets
3. Simplifies multi-device programming
4. Enables automatic backend selection

**Migration Path**:
1. Implement HBAL for current 4 models (CUDA, HIP, SYCL, OpenMP)
2. Convert 10 pilot benchmarks to HBAL
3. Validate performance parity
4. Incrementally migrate remaining benchmarks
5. Maintain model-specific versions for optimization studies

#### Strategy 2: Dynamic Kernel Generation

**Objective**: Adapt benchmarks to hardware capabilities at runtime

**Approach**: Template-based kernel specialization

```cpp
// Example: Matrix multiplication with runtime specialization
template<int TileM, int TileN, int TileK, typename Precision>
void gemm_kernel(/* parameters */);

// Runtime selection
void run_gemm_benchmark() {
  Device dev = hbal::get_device(0);

  // Query capabilities
  int max_threads = dev.query(DeviceQuery::MaxThreadsPerBlock);
  bool supports_fp8 = dev.query(DeviceQuery::SupportsFP8);
  bool has_tensor_cores = dev.query(DeviceQuery::HasTensorCores);

  // Select optimal kernel variant
  if (has_tensor_cores && supports_fp8) {
    gemm_kernel<16, 8, 32, fp8><<<grid, block>>>(/* args */);
  } else if (has_tensor_cores) {
    gemm_kernel<16, 8, 16, fp16><<<grid, block>>>(/* args */);
  } else {
    gemm_kernel<8, 8, 8, fp32><<<grid, block>>>(/* args */);
  }
}
```

**Techniques**:
1. **Capability queries**: Detect hardware features at runtime
2. **JIT compilation**: Generate specialized code on-the-fly (NVRTC, HIP RTC)
3. **Autotuning**: Benchmark multiple variants, select fastest
4. **Code caching**: Save tuned kernels for reuse

#### Strategy 3: Composable Benchmark Kernels

**Objective**: Build complex workloads from primitive operations

**Approach**: Kernel composition framework

```cpp
// Primitives
auto load = DataLoad<float, GlobalMemory>();
auto compute = GEMM<float, 16, 16, 16>();
auto reduce = Reduce<float, SumOp>();
auto store = DataStore<float, GlobalMemory>();

// Compose
auto fused_kernel = load >> compute >> reduce >> store;

// Execute
fused_kernel.run(input, output, config);

// Benefits:
// - Easy to create new benchmarks by combining primitives
// - Framework handles data movement optimization
// - Portable across hardware (primitives have per-device implementations)
```

**Primitive Library**:
- **Data movement**: Load, store, transpose, reshape
- **Arithmetic**: GEMM, elementwise ops, reductions
- **Communication**: AllReduce, broadcast, scatter/gather
- **Synchronization**: Barriers, atomics

#### Strategy 4: Multi-Device Orchestration

**Objective**: Support extreme-scale heterogeneous systems

**Approach**: Hierarchical device management

```cpp
// System topology
System sys = discover_system();
// sys.devices = [GPU0, GPU1, NPU0, NPU1, CPU0, CPU1, ...]

// Define execution plan
auto plan = ExecutionPlan::create()
  .add_stage("preprocess", CPU0, preprocess_kernel)
  .add_stage("compute", {GPU0, GPU1}, gemm_kernel, ParallelMode::DataParallel)
  .add_stage("postprocess", NPU0, quantize_kernel)
  .add_dependency("preprocess", "compute")
  .add_dependency("compute", "postprocess");

// Execute across heterogeneous devices
plan.run(input, output);
```

**Features**:
1. **Automatic data movement**: Framework manages transfers
2. **Load balancing**: Distribute work based on device capabilities
3. **Pipeline parallelism**: Overlap stages across devices
4. **Fault tolerance**: Checkpoint, retry on device failures

#### Strategy 5: Benchmarking Emerging Paradigms

**New Benchmark Categories**:

1. **Neuromorphic Computing**:
   - Spiking neural network kernels
   - Event-driven computation
   - Analog compute patterns

2. **Quantum-Classical Hybrid**:
   - Quantum circuit simulation
   - Variational quantum eigensolvers
   - Classical pre/post-processing for quantum

3. **Photonic Computing**:
   - Matrix-vector multiplication in optical domain
   - Interference-based computation

4. **Processing-in-Memory (PIM)**:
   - Near-memory compute patterns
   - Row-buffer operations
   - Reduce-in-place algorithms

**Implementation Strategy**:
- Start with **simulation/emulation** on current hardware
- Add **native implementations** as hardware becomes available
- Provide **performance models** for extrapolation

#### Strategy 6: Sustainability Metrics

**Objective**: Measure energy efficiency and carbon footprint

**New Metrics**:
1. **Performance per watt**: GFLOPS/W, GB/s/W
2. **Energy to solution**: Total joules to complete benchmark
3. **Carbon intensity**: gCO2e based on grid carbon intensity
4. **Data movement efficiency**: Bytes transferred / useful computation

**Integration**:
```cpp
// Enhanced benchmark output
Benchmark result = run_benchmark(jacobi);

std::cout << "Time: " << result.time << " ms\n";
std::cout << "Performance: " << result.gflops << " GFLOPS\n";
std::cout << "Energy: " << result.energy_joules << " J\n";
std::cout << "Efficiency: " << result.gflops_per_watt << " GFLOPS/W\n";
std::cout << "Carbon: " << result.carbon_gco2e << " gCO2e\n";
```

**Data Sources**:
- GPU power APIs (NVML, ROCm SMI)
- Grid carbon intensity APIs (Electricity Maps, WattTime)
- Hardware counters for data movement

### 5.3 Architectural Recommendations

#### Recommendation 1: Plugin Architecture for New Backends

**Design**:
```
HeCBench Core (HBAL)
├── Plugin API (stable interface)
└── Backends/
    ├── libhecbench_cuda.so (NVIDIA)
    ├── libhecbench_hip.so (AMD)
    ├── libhecbench_sycl.so (Intel, CPU)
    ├── libhecbench_metal.so (Apple)
    ├── libhecbench_npu.so (future: NPUs)
    └── libhecbench_custom.so (user-provided)
```

**Benefits**:
- Add new hardware without modifying core
- Third-party vendors can provide plugins
- Users can experiment with custom backends

#### Recommendation 2: Performance Model Database

**Purpose**: Predict benchmark performance on unseen hardware

**Approach**:
1. **Collect training data**: Run benchmarks on diverse hardware
2. **Extract features**: Device specs (cores, memory BW, etc.), benchmark characteristics
3. **Train models**: ML models to predict runtime
4. **Use for scheduling**: Select best device for each benchmark

**Schema**:
```sql
CREATE TABLE benchmark_runs (
  id INTEGER PRIMARY KEY,
  benchmark TEXT,
  device TEXT,
  runtime_ms REAL,
  energy_j REAL,
  timestamp TEXT
);

CREATE TABLE device_specs (
  device TEXT PRIMARY KEY,
  cores INTEGER,
  memory_gb REAL,
  bandwidth_gbs REAL,
  peak_flops REAL,
  architecture TEXT
);
```

**Query Example**:
```python
# Predict performance of "jacobi" on future GPU
prediction = model.predict(
  benchmark="jacobi",
  device_specs={"cores": 10000, "memory_gb": 96, "bandwidth_gbs": 3000}
)
print(f"Estimated runtime: {prediction.runtime_ms} ms")
```

#### Recommendation 3: Continuous Hardware Tracking

**Objective**: Maintain compatibility with evolving hardware

**Approach**:
1. **Hardware registry**: Database of supported devices
2. **Compatibility testing**: Automated tests on new GPUs
3. **Version matrix**: Track which benchmarks work on which hardware/compilers

**GitHub Actions Example**:
```yaml
strategy:
  matrix:
    include:
      - {gpu: nvidia-v100, arch: sm_70, compiler: nvcc-12.3}
      - {gpu: nvidia-a100, arch: sm_80, compiler: nvcc-12.3}
      - {gpu: nvidia-h100, arch: sm_90, compiler: nvcc-12.3}
      - {gpu: amd-mi100, arch: gfx908, compiler: hipcc-6.0}
      - {gpu: amd-mi250x, arch: gfx90a, compiler: hipcc-6.0}
      - {gpu: amd-mi300x, arch: gfx942, compiler: hipcc-6.0}
      - {gpu: intel-pvc, arch: pvc, compiler: icpx-2024.1}
```

#### Recommendation 4: Open Benchmark Registry

**Objective**: Community-contributed benchmarks

**Features**:
1. **Submission process**: PR-based with quality checks
2. **Peer review**: Community reviews new benchmarks
3. **Versioning**: Track benchmark versions over time
4. **Reproducibility**: Containerized environments for each benchmark

**Quality Criteria**:
- Multi-model implementation (≥3 of CUDA/HIP/SYCL/OpenMP/Serial)
- Correctness verification included
- Documentation (algorithm, expected performance, tuning notes)
- Passes automated tests

### 5.4 Timeline for Future-Proofing

**Year 1 (2025)**:
- Design HBAL abstraction layer
- Implement HBAL for existing 4 models
- Convert 50 pilot benchmarks to HBAL
- Add energy measurement infrastructure

**Year 2 (2026)**:
- Complete HBAL migration for all benchmarks
- Add plugin architecture for new backends
- Implement performance model database
- Add 20 new benchmarks (AI/ML, sparse, communication)

**Year 3 (2027)**:
- Add NPU/custom ASIC backend plugins
- Implement kernel composition framework
- Add 15 new benchmarks (emerging paradigms)
- Expand energy/sustainability metrics

**Year 4 (2028)**:
- Multi-device orchestration framework
- Processing-in-memory benchmark category
- Dynamic kernel generation for all benchmarks

**Year 5 (2029-2030)**:
- Neuromorphic/quantum benchmark categories
- Full heterogeneous system orchestration
- Community-driven benchmark registry operational

---

## Summary and Next Steps

### Phase Recap

| Phase | Focus | Key Deliverables | Estimated Effort |
|-------|-------|------------------|------------------|
| **1** | Review & Recommendations | Assessment report, plan.md | 1 week |
| **2** | CMake Build System | Unified build, presets, selective compilation | 4-6 weeks |
| **3** | Tooling & UX | Runner, visualization, containers, CI/CD | 8 weeks |
| **4** | Coverage Expansion | Serial implementations, 35-45 new benchmarks | 12 weeks |
| **5** | Future-Proofing | HBAL abstraction, plugin architecture, sustainability | Ongoing |

### Immediate Next Steps

**Week 1-2**:
1. Review and approve this plan
2. Set up project management (GitHub Projects, milestones)
3. Begin Phase 2: CMake proof-of-concept (10 benchmarks)

**Week 3-4**:
4. Complete CMake proof-of-concept
5. Design HBAL abstraction layer (Phase 5 foundation)
6. Start Phase 3: Enhanced runner tool

**Month 2**:
7. Migrate first category to CMake (Simulation benchmarks)
8. Implement campaign management in runner
9. Generate serial implementations from OpenMP (automated script)

### Success Metrics

**Phase 2 Success Criteria**:
- ✓ CMake builds all 508 benchmarks
- ✓ Selective building by name, category, model works
- ✓ Build time reduced by ≥30% vs. Make
- ✓ Presets for 8+ GPU architectures

**Phase 3 Success Criteria**:
- ✓ Campaign execution across multiple GPUs
- ✓ Result database with historical tracking
- ✓ Automated visualization (HTML reports)
- ✓ Containers for CUDA, HIP, SYCL

**Phase 4 Success Criteria**:
- ✓ Serial implementation for all 508 benchmarks
- ✓ 35+ new benchmarks added (AI/ML, sparse, communication)
- ✓ Quad-model coverage ≥95%

**Phase 5 Success Criteria**:
- ✓ HBAL abstraction layer operational
- ✓ Plugin architecture with ≥1 third-party backend
- ✓ Energy metrics for all benchmarks
- ✓ Performance model database with predictions

---

## Appendix: Risk Analysis

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CMake migration breaks existing workflows | Medium | High | Maintain Makefiles in parallel for 6-12 months |
| Compiler compatibility issues with new build system | High | Medium | Extensive testing matrix in CI/CD |
| Performance regression in CMake builds | Low | Medium | Benchmark build times before/after |
| HBAL abstraction overhead | Medium | High | Extensive performance validation, optional per-benchmark |
| New benchmarks too hardware-specific | Medium | Medium | Require ≥3 model implementations for acceptance |
| Community adoption of new tools | Medium | High | Comprehensive documentation, tutorials |

### Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient GPU hardware for testing | Low | High | Cloud GPU resources (AWS, Azure), partnerships |
| Limited development resources | Medium | High | Phased approach, prioritize high-impact items |
| Compiler/SDK breaking changes | High | Medium | Pin versions, test beta releases early |

### Mitigation Strategy

1. **Incremental migration**: Never break existing functionality
2. **Extensive testing**: Automated CI/CD on multiple hardware platforms
3. **Community engagement**: Solicit feedback early and often
4. **Flexibility**: Be prepared to adjust plan based on lessons learned

---

**Plan Prepared By**: Claude (Anthropic)
**Date**: 2025-12-06
**Version**: 1.0
**Status**: Ready for review and approval
