# blas-zgemm-sycl/CMakeLists.txt

# Intel targets use oneMKL. NVIDIA and AMD targets use oneMath, whose cuBLAS or
# rocBLAS backend must support gemm_batch with INT8 inputs and INT32 output
# (https://github.com/uxlfoundation/oneMath/pull/803 for cuBLAS,
#  https://github.com/uxlfoundation/oneMath/pull/804 for rocBLAS).
set(HECBENCH_ONEMATH_ROOT "" CACHE PATH "oneMath install prefix for SYCL on NVIDIA and AMD")

if(HECBENCH_SYCL_TARGET MATCHES "nvptx|amdgcn")
    if(NOT HECBENCH_ONEMATH_ROOT)
        message(STATUS "Skipping blas-zgemm-sycl (set HECBENCH_ONEMATH_ROOT)")
        return()
    endif()
    if(HECBENCH_SYCL_TARGET MATCHES "nvptx")
        set(BLAS_ZGEMM_ONEMATH_BACKEND onemath_blas_cublas)
    else()
        set(BLAS_ZGEMM_ONEMATH_BACKEND onemath_blas_rocblas)
    endif()
    set(BLAS_ZGEMM_COMPILE_OPTIONS -DUSE_ONEMATH)
    set(BLAS_ZGEMM_INCLUDE_DIRS ${HECBENCH_ONEMATH_ROOT}/include)
    set(BLAS_ZGEMM_LINK_LIBRARIES
        -L${HECBENCH_ONEMATH_ROOT}/lib -Wl,-rpath,${HECBENCH_ONEMATH_ROOT}/lib
        -lonemath -l${BLAS_ZGEMM_ONEMATH_BACKEND})
else()
    set(BLAS_ZGEMM_COMPILE_OPTIONS "")
    set(BLAS_ZGEMM_INCLUDE_DIRS "")
    set(BLAS_ZGEMM_LINK_LIBRARIES -qmkl=parallel)
endif()

add_hecbench_benchmark(
    NAME blas-zgemm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    COMPILE_OPTIONS ${BLAS_ZGEMM_COMPILE_OPTIONS}
    INCLUDE_DIRS ${BLAS_ZGEMM_INCLUDE_DIRS}
    LINK_LIBRARIES ${BLAS_ZGEMM_LINK_LIBRARIES}
    TEST_ARGS 256 256 256 2
    TEST_REGEX "PASS"
    TEST_TIMEOUT 300
)
