# dwconv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dwconv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
