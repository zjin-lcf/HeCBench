# dxtc2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dxtc2
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
