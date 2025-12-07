# mcmd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcmd
    MODEL hip
    SOURCES force_kernel.cu
    CATEGORIES algorithms
)
