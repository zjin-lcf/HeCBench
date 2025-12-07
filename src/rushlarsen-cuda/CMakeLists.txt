# rushlarsen-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rushlarsen
    MODEL cuda
    SOURCES kernels.cu main.cu reference.cu utils.cu
    CATEGORIES algorithms
)
