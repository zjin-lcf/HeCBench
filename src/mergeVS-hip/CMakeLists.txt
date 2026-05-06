# mergeVS-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mergeVS
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
