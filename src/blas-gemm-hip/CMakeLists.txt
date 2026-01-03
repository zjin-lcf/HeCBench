# blas-gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    COMPILE_OPTIONS -DHIPBLAS_USE_HIP_HALF 
    LINK_LIBRARIES hipblas
)
