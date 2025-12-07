# svd3x3-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME svd3x3
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
