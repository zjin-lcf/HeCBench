# spsort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spsort
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
