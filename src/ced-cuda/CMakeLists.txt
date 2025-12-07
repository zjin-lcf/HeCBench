# ced-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ced
    MODEL cuda
    SOURCES kernel_reference.cu main.cpp main.cu
    CATEGORIES algorithms
)
