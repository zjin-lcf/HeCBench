# scatter-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatter
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
