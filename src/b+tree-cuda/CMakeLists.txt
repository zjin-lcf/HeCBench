# b+tree-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME b+tree
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
