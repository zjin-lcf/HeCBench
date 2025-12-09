# streamUM-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamUM
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
