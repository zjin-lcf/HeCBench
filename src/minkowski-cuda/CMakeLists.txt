# minkowski-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME minkowski
    MODEL cuda
    SOURCES main.cu verify.cpp
    CATEGORIES algorithms
)
