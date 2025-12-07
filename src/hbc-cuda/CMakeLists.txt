# hbc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hbc
    MODEL cuda
    SOURCES kernels.cu main.cpp parse.cpp sequential.cpp util.cpp
    CATEGORIES algorithms
)
