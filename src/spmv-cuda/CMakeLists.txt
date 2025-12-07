# spmv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spmv
    MODEL cuda
    SOURCES kernels.cu main.cpp utils.cpp
    CATEGORIES algorithms
)
