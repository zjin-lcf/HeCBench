# dslash-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dslash
    MODEL hip
    SOURCES dslash.cu kernels.cu main.cu
    CATEGORIES algorithms
)
