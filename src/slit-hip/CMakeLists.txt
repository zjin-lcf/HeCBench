# slit-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME slit
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
