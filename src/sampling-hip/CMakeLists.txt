# sampling-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sampling
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
