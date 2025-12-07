# dwconv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dwconv
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
