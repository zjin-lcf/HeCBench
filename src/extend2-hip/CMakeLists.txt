# extend2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME extend2
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
