# blas-gemmStridedBatched-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmStridedBatched
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
