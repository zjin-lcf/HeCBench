# mmcsf-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mmcsf
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
