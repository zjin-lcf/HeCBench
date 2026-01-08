# clink-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME clink
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
