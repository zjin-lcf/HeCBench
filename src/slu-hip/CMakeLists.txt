# slu-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME slu
    MODEL hip
    SOURCES numeric.cu
    CATEGORIES algorithms
)
