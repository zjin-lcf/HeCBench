# lanczos-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lanczos
    MODEL cuda
    SOURCES lanczos.cu main.cu
    CATEGORIES algorithms
)
