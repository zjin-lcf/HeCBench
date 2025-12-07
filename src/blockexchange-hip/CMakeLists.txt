# blockexchange-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockexchange
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
