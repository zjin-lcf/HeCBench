# src/CMakeLists.txt
# Benchmark discovery and registration

# Add common include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# List of benchmarks to build (will be expanded over time)
# For now, we start with a proof-of-concept set
set(HECBENCH_POC_BENCHMARKS
    jacobi
    nbody
    softmax
    attention
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
