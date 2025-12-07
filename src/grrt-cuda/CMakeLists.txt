# grrt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
