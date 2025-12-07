# hexciton-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hexciton
    MODEL cuda
    SOURCES kernels.cu main.cu reference.cu utils.cu
    CATEGORIES algorithms
)
