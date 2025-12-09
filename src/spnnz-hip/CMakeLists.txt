# spnnz-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spnnz
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
