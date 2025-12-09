# lavaMD-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lavaMD
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
