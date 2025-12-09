# hypterm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hypterm
    MODEL hip
    SOURCES kernels.cu
    CATEGORIES algorithms
)
