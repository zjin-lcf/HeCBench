# CMake Build System for HeCBench

This document describes the new CMake-based build system for HeCBench, which replaces the previous Makefile-based approach.

## Overview

The CMake build system provides:
- **Unified configuration** via CMake presets for common GPU architectures
- **Selective building** by benchmark, programming model, or category
- **Automatic compiler detection** for CUDA, HIP, SYCL, and OpenMP
- **Parallel builds** across benchmarks
- **IDE integration** (CLion, VS Code, etc.)

## Quick Start

### Prerequisites

Depending on which programming models you want to build:

- **CUDA**: NVIDIA CUDA Toolkit (11.0+)
- **HIP**: AMD ROCm (5.0+)
- **SYCL**: Intel oneAPI DPC++ or hipSYCL
- **OpenMP**: Intel oneAPI, NVIDIA HPC SDK, or AOMP
- **CMake**: 3.21 or later
- **Ninja** (recommended) or Make

### Build with a Preset

The simplest way to build is using a CMake preset:

```bash
# List available presets
cmake --list-presets

# Configure for NVIDIA A100 (sm_80)
cmake --preset cuda-sm80

# Build all configured benchmarks
cmake --build build/cuda-sm80

# Or build with Ninja in parallel
cmake --build build/cuda-sm80 --parallel
```

### Available Presets

#### NVIDIA GPUs (CUDA)
- `cuda-sm60` - Pascal (GTX 1080, P100)
- `cuda-sm70` - Volta (V100)
- `cuda-sm80` - Ampere (A100)
- `cuda-sm90` - Hopper (H100, H200)

#### AMD GPUs (HIP)
- `hip-gfx908` - MI100
- `hip-gfx90a` - MI250X
- `hip-gfx942` - MI300A/X

#### SYCL
- `sycl-cuda` - SYCL with CUDA backend
- `sycl-hip` - SYCL with HIP backend
- `sycl-cpu` - SYCL with CPU backend

#### OpenMP
- `openmp-intel` - Intel compiler with OpenMP offload
- `openmp-nvidia` - NVIDIA nvc++ with OpenMP offload

#### Multi-Model
- `all-models` - Build all programming models (requires all compilers)

## Building Specific Benchmarks

### Build a Single Benchmark (All Models)

```bash
cmake --preset cuda-sm80
cmake --build build/cuda-sm80 --target jacobi-all
```

This builds jacobi for all enabled models (if you used `all-models` preset).

### Build a Specific Model Variant

```bash
# Build only CUDA version of jacobi
cmake --build build/cuda-sm80 --target jacobi-cuda

# Build only HIP version of attention
cmake --preset hip-gfx90a
cmake --build build/hip-gfx90a --target attention-hip
```

### Build by Category

```bash
# Build all machine learning benchmarks
cmake --build build/cuda-sm80 --target category-ml

# Build all graph benchmarks
cmake --build build/cuda-sm80 --target category-graph

# Build all simulation benchmarks
cmake --build build/cuda-sm80 --target category-simulation
```

## Advanced Configuration

### Custom Configuration

If you need more control, configure without a preset:

```bash
cmake -B build/custom \
  -G Ninja \
  -DHECBENCH_ENABLE_CUDA=ON \
  -DHECBENCH_ENABLE_HIP=OFF \
  -DHECBENCH_ENABLE_SYCL=OFF \
  -DHECBENCH_ENABLE_OPENMP=OFF \
  -DHECBENCH_CUDA_ARCH=80

cmake --build build/custom
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HECBENCH_ENABLE_CUDA` | ON | Enable CUDA benchmarks |
| `HECBENCH_ENABLE_HIP` | ON | Enable HIP benchmarks |
| `HECBENCH_ENABLE_SYCL` | ON | Enable SYCL benchmarks |
| `HECBENCH_ENABLE_OPENMP` | ON | Enable OpenMP benchmarks |
| `HECBENCH_CUDA_ARCH` | sm_80 | CUDA architecture (60, 70, 80, 90, etc.) |
| `HECBENCH_HIP_ARCH` | gfx90a | HIP architecture (gfx908, gfx90a, gfx942, etc.) |
| `HECBENCH_SYCL_TARGET` | (auto) | SYCL target backend |
| `HECBENCH_ENABLE_TESTING` | ON | Enable testing support |
| `HECBENCH_BUILD_ALL_BENCHMARKS` | ON | Build all vs. selective |

### Multi-Architecture Builds

To build for multiple GPU architectures, use multiple configure+build cycles:

```bash
# Build for A100
cmake --preset cuda-sm80
cmake --build build/cuda-sm80

# Build for V100
cmake --preset cuda-sm70
cmake --build build/cuda-sm70

# Build for MI250X
cmake --preset hip-gfx90a
cmake --build build/hip-gfx90a
```

## Output Structure

Compiled binaries are placed in:

```
build/<preset>/bin/<model>/
├── cuda/
│   ├── jacobi
│   ├── bfs
│   ├── softmax
│   └── attention
├── hip/
│   └── ...
├── sycl/
│   └── ...
└── omp/
    └── ...
```

## Running Benchmarks

```bash
# Run a benchmark directly
./build/cuda-sm80/bin/cuda/jacobi

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 ./build/cuda-sm80/bin/cuda/attention
```

## Current Status (Proof of Concept)

The CMake build system currently supports **4 benchmarks** (16 implementations):

| Benchmark | CUDA | HIP | SYCL | OpenMP | Categories |
|-----------|------|-----|------|--------|------------|
| jacobi | ✓ | ✓ | ✓ | ✓ | simulation, math |
| bfs | ✓ | ✓ | ✓ | ✓ | graph, algorithms |
| softmax | ✓ | ✓ | ✓ | ✓ | ml, math |
| attention | ✓ | ✓ | ✓ | ✓ | ml |

### Migration Status

- **Phase 1**: ✅ Infrastructure complete
- **Phase 2**: 🔄 Proof of concept (4/508 benchmarks)
- **Phase 3**: ⏳ Full migration pending

## Adding New Benchmarks

To convert a benchmark to CMake, add a `CMakeLists.txt` in each model directory:

```cmake
# src/mybench-cuda/CMakeLists.txt
add_hecbench_benchmark(
    NAME mybench
    MODEL cuda
    SOURCES main.cu kernel.cu
    CATEGORIES simulation physics
)
```

Then add the benchmark name to `src/CMakeLists.txt`:

```cmake
set(HECBENCH_POC_BENCHMARKS
    jacobi
    bfs
    softmax
    attention
    mybench  # <-- Add here
)
```

## Troubleshooting

### Compiler Not Found

```
CMake Error: Could not find CUDA/HIP/SYCL compiler
```

**Solution**: Install the required compiler or disable that model:
```bash
cmake --preset cuda-sm80 -DHECBENCH_ENABLE_HIP=OFF
```

### Architecture Mismatch

```
Error: Unsupported architecture sm_XX
```

**Solution**: Use a preset matching your GPU or set the architecture manually:
```bash
cmake --preset cuda-sm80 -DHECBENCH_CUDA_ARCH=86
```

### Missing Dependencies

Some benchmarks may require additional libraries (oneDPL, TBB, etc.). These will be detected automatically if present.

## IDE Integration

### Visual Studio Code

Install the CMake Tools extension, then:
1. Open the HeCBench folder
2. Select a CMake preset from the status bar
3. Click "Build" or press F7

### CLion

CLion automatically detects `CMakePresets.json`:
1. Open the HeCBench project
2. CLion will import presets automatically
3. Select a profile from the dropdown
4. Build → Build Project

## Comparison with Makefile Build

| Feature | Makefile | CMake |
|---------|----------|-------|
| Configuration | Edit 2,587 individual Makefiles | Single preset selection |
| Parallel builds | Per-benchmark only | Across all benchmarks |
| Selective building | Manual (cd + make) | Target-based (by name/category) |
| IDE support | Limited | Full integration |
| Multi-arch | Rebuild everything | Separate build dirs |
| Dependency tracking | Manual | Automatic |

## Future Enhancements

Planned improvements:
- [ ] Migrate remaining 504 benchmarks
- [ ] CTest integration for automated testing
- [ ] CPack support for distribution
- [ ] Benchmark performance regression tracking
- [ ] Docker container presets
- [ ] GitHub Actions CI integration

## Getting Help

- Report issues: https://github.com/anthropics/hecbench/issues
- Main README: [README.md](README.md)
- Full renovation plan: [plan.md](plan.md)

---

**Last Updated**: 2025-12-06
**Status**: Proof of Concept (Phase 2)
