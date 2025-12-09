# diamond-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME diamond
    MODEL hip
    SOURCES masking.cu
    CATEGORIES algorithms
)
