# BenchmarkMacros.cmake
# Macros and functions for registering HeCBench benchmarks

# Global list to track all registered benchmarks
set_property(GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS "")
set_property(GLOBAL PROPERTY HECBENCH_CATEGORIES "")

# add_hecbench_benchmark
#
# Register a HeCBench benchmark
#
# Arguments:
#   NAME        - Benchmark name (e.g., "jacobi")
#   MODEL       - Programming model: cuda, hip, sycl, omp
#   SOURCES     - Source files
#   CATEGORIES  - Categories (e.g., simulation, math)
#   COMPILE_OPTIONS - Additional compile options (optional)
#   LINK_LIBRARIES  - Additional libraries to link (optional)
#
# Example:
#   add_hecbench_benchmark(
#       NAME jacobi
#       MODEL cuda
#       SOURCES main.cu
#       CATEGORIES simulation math
#   )
#
function(add_hecbench_benchmark)
    # Parse arguments
    set(options "")
    set(oneValueArgs NAME MODEL)
    set(multiValueArgs SOURCES CATEGORIES COMPILE_OPTIONS LINK_LIBRARIES INCLUDE_DIRS)
    cmake_parse_arguments(BENCH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT BENCH_NAME)
        message(FATAL_ERROR "add_hecbench_benchmark: NAME is required")
    endif()
    if(NOT BENCH_MODEL)
        message(FATAL_ERROR "add_hecbench_benchmark: MODEL is required")
    endif()
    if(NOT BENCH_SOURCES)
        message(FATAL_ERROR "add_hecbench_benchmark: SOURCES is required")
    endif()

    # Normalize model name
    string(TOLOWER "${BENCH_MODEL}" BENCH_MODEL_LOWER)

    # Check if this model is enabled
    set(MODEL_ENABLED FALSE)
    if(BENCH_MODEL_LOWER STREQUAL "cuda" AND HECBENCH_ENABLE_CUDA)
        set(MODEL_ENABLED TRUE)
    elseif(BENCH_MODEL_LOWER STREQUAL "hip" AND HECBENCH_ENABLE_HIP)
        set(MODEL_ENABLED TRUE)
    elseif(BENCH_MODEL_LOWER STREQUAL "sycl" AND HECBENCH_ENABLE_SYCL)
        set(MODEL_ENABLED TRUE)
    elseif((BENCH_MODEL_LOWER STREQUAL "omp" OR BENCH_MODEL_LOWER STREQUAL "openmp") AND HECBENCH_ENABLE_OPENMP)
        set(MODEL_ENABLED TRUE)
    endif()

    if(NOT MODEL_ENABLED)
        message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER} (model not enabled)")
        return()
    endif()

    # Create target name
    set(TARGET_NAME "${BENCH_NAME}-${BENCH_MODEL_LOWER}")

    # Add executable
    add_executable(${TARGET_NAME} ${BENCH_SOURCES})

    # Set target properties
    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${BENCH_NAME}
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${BENCH_MODEL_LOWER}"
    )

    # Add include directories
    if(BENCH_INCLUDE_DIRS)
        target_include_directories(${TARGET_NAME} PRIVATE ${BENCH_INCLUDE_DIRS})
    endif()

    # Add common include directory
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/src/include")

    # Model-specific configuration
    if(BENCH_MODEL_LOWER STREQUAL "cuda")
        # CUDA configuration
        set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart)

    elseif(BENCH_MODEL_LOWER STREQUAL "hip")
        # HIP configuration
        target_compile_options(${TARGET_NAME} PRIVATE
            -std=c++17
            --offload-arch=${HECBENCH_HIP_ARCH}
        )
        set_target_properties(${TARGET_NAME} PROPERTIES
            LINKER_LANGUAGE CXX
        )
        target_link_libraries(${TARGET_NAME} PRIVATE hip::host)

    elseif(BENCH_MODEL_LOWER STREQUAL "sycl")
        # SYCL configuration
        target_compile_options(${TARGET_NAME} PRIVATE
            -fsycl
            ${SYCL_FLAGS}
        )
        target_link_options(${TARGET_NAME} PRIVATE
            -fsycl
            ${SYCL_FLAGS}
        )

    elseif(BENCH_MODEL_LOWER STREQUAL "omp" OR BENCH_MODEL_LOWER STREQUAL "openmp")
        # OpenMP configuration
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Add user-specified compile options
    if(BENCH_COMPILE_OPTIONS)
        target_compile_options(${TARGET_NAME} PRIVATE ${BENCH_COMPILE_OPTIONS})
    endif()

    # Add user-specified link libraries
    if(BENCH_LINK_LIBRARIES)
        target_link_libraries(${TARGET_NAME} PRIVATE ${BENCH_LINK_LIBRARIES})
    endif()

    # Register benchmark in global list
    get_property(ALL_BENCHMARKS GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS)
    list(APPEND ALL_BENCHMARKS ${TARGET_NAME})
    set_property(GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS "${ALL_BENCHMARKS}")

    # Register categories
    foreach(CATEGORY ${BENCH_CATEGORIES})
        string(TOLOWER "${CATEGORY}" CATEGORY_LOWER)
        get_property(CAT_BENCHMARKS GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY_LOWER})
        list(APPEND CAT_BENCHMARKS ${TARGET_NAME})
        set_property(GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY_LOWER} "${CAT_BENCHMARKS}")

        # Track unique categories
        get_property(ALL_CATEGORIES GLOBAL PROPERTY HECBENCH_CATEGORIES)
        if(NOT CATEGORY_LOWER IN_LIST ALL_CATEGORIES)
            list(APPEND ALL_CATEGORIES ${CATEGORY_LOWER})
            set_property(GLOBAL PROPERTY HECBENCH_CATEGORIES "${ALL_CATEGORIES}")
        endif()
    endforeach()

    # Create an alias target for just this benchmark (all models)
    if(NOT TARGET ${BENCH_NAME}-all)
        add_custom_target(${BENCH_NAME}-all)
    endif()
    add_dependencies(${BENCH_NAME}-all ${TARGET_NAME})

    message(STATUS "Registered benchmark: ${TARGET_NAME}")
endfunction()

# create_category_targets
#
# Create convenience targets for building benchmarks by category
# Should be called after all benchmarks are registered
#
function(create_category_targets)
    get_property(ALL_CATEGORIES GLOBAL PROPERTY HECBENCH_CATEGORIES)

    foreach(CATEGORY ${ALL_CATEGORIES})
        get_property(CAT_BENCHMARKS GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY})

        if(CAT_BENCHMARKS)
            # Create target for this category
            add_custom_target(category-${CATEGORY})
            add_dependencies(category-${CATEGORY} ${CAT_BENCHMARKS})
            message(STATUS "Created category target: category-${CATEGORY} (${list(LENGTH CAT_BENCHMARKS)} benchmarks)")
        endif()
    endforeach()
endfunction()

# print_benchmark_summary
#
# Print summary of registered benchmarks
#
function(print_benchmark_summary)
    get_property(ALL_BENCHMARKS GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS)
    list(LENGTH ALL_BENCHMARKS BENCHMARK_COUNT)

    message(STATUS "")
    message(STATUS "=== Benchmark Registration Summary ===")
    message(STATUS "Total benchmarks registered: ${BENCHMARK_COUNT}")

    get_property(ALL_CATEGORIES GLOBAL PROPERTY HECBENCH_CATEGORIES)
    foreach(CATEGORY ${ALL_CATEGORIES})
        get_property(CAT_BENCHMARKS GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY})
        list(LENGTH CAT_BENCHMARKS CAT_COUNT)
        message(STATUS "  ${CATEGORY}: ${CAT_COUNT}")
    endforeach()

    message(STATUS "======================================")
    message(STATUS "")
endfunction()
