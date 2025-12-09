# nw-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nw
    MODEL cuda
    SOURCES nw.cu reference.cpp
    CATEGORIES algorithms
)
