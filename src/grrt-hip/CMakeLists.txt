# grrt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
