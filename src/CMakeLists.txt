# src/CMakeLists.txt
# Benchmark discovery and registration

# Add common include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# List of benchmarks to build (will be expanded over time)
# Started with proof-of-concept, now expanding coverage
set(HECBENCH_POC_BENCHMARKS
    # Original POC
    jacobi
    bfs
    softmax
    attention
    # Batch 1: A-F alphabetically (15 benchmarks)
    accuracy
    ace
    adam
    adamw
    aes
    affine
    aidw
    amgmk
    aobench
    backprop
    bilateral
    cfd
    clenergy
    dct8x8
    fft
)

# Include benchmark subdirectories
foreach(BENCHMARK ${HECBENCH_POC_BENCHMARKS})
    # Check which models exist for this benchmark
    foreach(MODEL cuda hip sycl omp)
        set(BENCHMARK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${BENCHMARK}-${MODEL}")
        if(EXISTS "${BENCHMARK_DIR}/CMakeLists.txt")
            add_subdirectory(${BENCHMARK_DIR})
        endif()
    endforeach()
endforeach()

# Create category targets after all benchmarks are registered
create_category_targets()

# Print summary
print_benchmark_summary()
