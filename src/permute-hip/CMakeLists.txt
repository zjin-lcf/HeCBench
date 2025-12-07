# permute-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME permute
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
