# gd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gd
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
