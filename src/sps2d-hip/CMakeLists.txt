# sps2d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sps2d
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
