# sobel-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sobel
    MODEL cuda
    SOURCES kernels.cu main.cu reference.cu
    CATEGORIES algorithms
)
