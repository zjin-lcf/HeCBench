# hbc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hbc
    MODEL hip
    SOURCES kernels.cu
    CATEGORIES algorithms
)
