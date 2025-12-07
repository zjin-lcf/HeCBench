# bn-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
