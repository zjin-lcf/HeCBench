# flame-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME flame
    MODEL cuda
    SOURCES iteration.cu kernels.cu main.cu utils.cu
    CATEGORIES algorithms
)
