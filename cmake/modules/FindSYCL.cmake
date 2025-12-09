# FindSYCL.cmake
# Find SYCL compiler and libraries
#
# This module defines:
#  SYCL_FOUND       - True if SYCL was found
#  SYCL_COMPILER    - Path to SYCL-capable compiler
#  SYCL_VERSION     - SYCL compiler version
#  SYCL_FLAGS       - Compilation flags for SYCL

if(SYCL_FOUND)
    return()
endif()

# SYCL can be provided by multiple compilers:
# - Intel DPC++ (icpx, clang++)
# - hipSYCL (syclcc, clang++)
# - ComputeCpp (compute++)

set(SYCL_FLAGS "")

# Try to find Intel DPC++
find_program(SYCL_COMPILER
    NAMES icpx clang++
    PATHS
        ENV ONEAPI_ROOT
        ENV DPCPP_HOME
        /opt/intel/oneapi/compiler/latest/linux
    PATH_SUFFIXES bin
    DOC "SYCL compiler"
)

if(SYCL_COMPILER)
    # Check if compiler supports SYCL
    execute_process(
        COMMAND ${SYCL_COMPILER} -fsycl --version
        OUTPUT_VARIABLE SYCL_VERSION_OUTPUT
        ERROR_VARIABLE SYCL_VERSION_ERROR
        RESULT_VARIABLE SYCL_VERSION_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(SYCL_VERSION_RESULT EQUAL 0)
        # Compiler supports -fsycl flag
        set(SYCL_FLAGS "-fsycl")

        # Parse version
        if(SYCL_VERSION_OUTPUT MATCHES "DPC\\+\\+/C\\+\\+ Compiler ([0-9]+\\.[0-9]+\\.[0-9]+)")
            set(SYCL_VERSION "${CMAKE_MATCH_1}")
            set(SYCL_COMPILER_TYPE "Intel DPC++")
        elseif(SYCL_VERSION_OUTPUT MATCHES "clang version ([0-9]+\\.[0-9]+\\.[0-9]+)")
            set(SYCL_VERSION "${CMAKE_MATCH_1}")
            set(SYCL_COMPILER_TYPE "Clang with SYCL")
        endif()

        # Detect backend support
        # Check for CUDA backend
        execute_process(
            COMMAND ${SYCL_COMPILER} -fsycl -fsycl-targets=nvptx64-nvidia-cuda --version
            RESULT_VARIABLE SYCL_CUDA_RESULT
            OUTPUT_QUIET ERROR_QUIET
        )
        if(SYCL_CUDA_RESULT EQUAL 0)
            set(SYCL_SUPPORTS_CUDA TRUE)
        endif()

        # Check for HIP backend
        execute_process(
            COMMAND ${SYCL_COMPILER} -fsycl -fsycl-targets=amdgcn-amd-amdhsa --version
            RESULT_VARIABLE SYCL_HIP_RESULT
            OUTPUT_QUIET ERROR_QUIET
        )
        if(SYCL_HIP_RESULT EQUAL 0)
            set(SYCL_SUPPORTS_HIP TRUE)
        endif()

    else()
        # Try hipSYCL
        find_program(HIPSYCL_COMPILER
            NAMES syclcc
            PATHS
                ENV HIPSYCL_HOME
            PATH_SUFFIXES bin
        )

        if(HIPSYCL_COMPILER)
            set(SYCL_COMPILER ${HIPSYCL_COMPILER})
            set(SYCL_COMPILER_TYPE "hipSYCL")

            execute_process(
                COMMAND ${SYCL_COMPILER} --version
                OUTPUT_VARIABLE SYCL_VERSION_OUTPUT
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if(SYCL_VERSION_OUTPUT MATCHES "hipSYCL version ([0-9]+\\.[0-9]+\\.[0-9]+)")
                set(SYCL_VERSION "${CMAKE_MATCH_1}")
            endif()
        endif()
    endif()
endif()

# Set additional flags based on cache variables
if(DEFINED HECBENCH_SYCL_TARGET)
    list(APPEND SYCL_FLAGS "-fsycl-targets=${HECBENCH_SYCL_TARGET}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SYCL
    REQUIRED_VARS SYCL_COMPILER
    VERSION_VAR SYCL_VERSION
)

if(SYCL_FOUND)
    message(STATUS "SYCL compiler type: ${SYCL_COMPILER_TYPE}")
    if(SYCL_SUPPORTS_CUDA)
        message(STATUS "  CUDA backend: supported")
    endif()
    if(SYCL_SUPPORTS_HIP)
        message(STATUS "  HIP backend: supported")
    endif()
endif()

mark_as_advanced(
    SYCL_COMPILER
    SYCL_VERSION
    SYCL_FLAGS
    SYCL_COMPILER_TYPE
)
