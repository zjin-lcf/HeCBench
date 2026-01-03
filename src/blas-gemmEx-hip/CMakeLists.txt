# blas-gemmEx-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmEx
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas
)
