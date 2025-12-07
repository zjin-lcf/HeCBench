# scan3-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan3
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
