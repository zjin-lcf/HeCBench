# blas-gemmBatched-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmBatched
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas
)
