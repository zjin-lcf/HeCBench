# sampling-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sampling
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
