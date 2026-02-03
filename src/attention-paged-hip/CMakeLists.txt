# attention-paged-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME attention-paged
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
