# bn-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
