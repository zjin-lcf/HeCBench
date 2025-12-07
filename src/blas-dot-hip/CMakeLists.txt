# blas-dot-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-dot
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
