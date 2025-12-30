# f8cast-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME f8cast
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
