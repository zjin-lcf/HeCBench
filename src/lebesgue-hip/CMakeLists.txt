# lebesgue-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lebesgue
    MODEL hip
    SOURCES kernels.cu
    CATEGORIES algorithms
)
