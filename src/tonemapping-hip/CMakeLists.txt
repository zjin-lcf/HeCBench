# tonemapping-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tonemapping
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
