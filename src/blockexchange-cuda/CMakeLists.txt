# blockexchange-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockexchange
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
