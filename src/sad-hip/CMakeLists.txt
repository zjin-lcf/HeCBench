# sad-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sad
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
