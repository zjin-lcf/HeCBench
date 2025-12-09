# deredundancy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL hip
    SOURCES kernels.cu main.cu utils.cu
    CATEGORIES algorithms
)
