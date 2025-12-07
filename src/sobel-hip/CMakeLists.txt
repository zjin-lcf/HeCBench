# sobel-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sobel
    MODEL hip
    SOURCES kernels.cu main.cu reference.cu
    CATEGORIES algorithms
)
