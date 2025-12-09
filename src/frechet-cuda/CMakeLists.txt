# frechet-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME frechet
    MODEL cuda
    SOURCES main.cpp main.cu
    CATEGORIES algorithms
)
