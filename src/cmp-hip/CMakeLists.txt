# cmp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cmp
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
