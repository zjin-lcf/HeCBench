# streamUM-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamUM
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
