# interleave-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME interleave
    MODEL cuda
    SOURCES main.cu util.cpp
    CATEGORIES algorithms
)
