# blas-gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
