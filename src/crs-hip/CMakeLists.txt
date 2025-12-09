# crs-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME crs
    MODEL hip
    SOURCES kernels.cu main.cu utils.cu
    CATEGORIES algorithms
)
