# permute-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME permute
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
