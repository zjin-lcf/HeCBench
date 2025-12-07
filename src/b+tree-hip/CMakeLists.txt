# b+tree-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME b+tree
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
