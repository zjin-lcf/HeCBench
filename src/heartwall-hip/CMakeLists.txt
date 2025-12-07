# heartwall-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME heartwall
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
