# warpexchange-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpexchange
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
