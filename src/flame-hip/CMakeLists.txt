# flame-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME flame
    MODEL hip
    SOURCES iteration.cu kernels.cu main.cu utils.cu
    CATEGORIES algorithms
)
