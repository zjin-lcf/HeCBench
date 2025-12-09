# warpsort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpsort
    MODEL cuda
    SOURCES main.cpp warpsort.cu
    CATEGORIES algorithms
)
