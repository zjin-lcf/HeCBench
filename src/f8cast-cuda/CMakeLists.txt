# f8cast-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME f8cast
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
