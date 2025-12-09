# axhelm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME axhelm
    MODEL cuda
    SOURCES axhelmKernel.cpp axhelmReference.cpp main.cu meshBasis.cpp
    CATEGORIES algorithms
)
