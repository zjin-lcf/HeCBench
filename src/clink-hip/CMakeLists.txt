# clink-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME clink
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
