# svd3x3-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME svd3x3
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
