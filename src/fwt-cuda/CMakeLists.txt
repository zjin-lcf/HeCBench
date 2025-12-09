# fwt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fwt
    MODEL cuda
    SOURCES kernels.cu main.cu reference.cu
    CATEGORIES algorithms
)
