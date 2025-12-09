# tsp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
