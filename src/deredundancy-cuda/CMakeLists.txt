# deredundancy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL cuda
    SOURCES kernels.cu main.cu utils.cu
    CATEGORIES algorithms
)
