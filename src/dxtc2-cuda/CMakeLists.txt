# dxtc2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dxtc2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
