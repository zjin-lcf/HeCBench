# meanshift-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME meanshift
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
