# urng-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME urng
    MODEL cuda
    SOURCES kernel.cu main.cu
    CATEGORIES algorithms
)
