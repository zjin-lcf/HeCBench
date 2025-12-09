# reaction-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME reaction
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
