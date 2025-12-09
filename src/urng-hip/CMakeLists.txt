# urng-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME urng
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
