# fwt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fwt
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
