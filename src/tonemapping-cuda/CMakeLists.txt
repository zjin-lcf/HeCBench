# tonemapping-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tonemapping
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
