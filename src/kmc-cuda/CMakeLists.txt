# kmc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmc
    MODEL cuda
    SOURCES main.cpp svm.cpp
    CATEGORIES algorithms
)
