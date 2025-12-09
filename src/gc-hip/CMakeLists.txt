# gc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gc
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
