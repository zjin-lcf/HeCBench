# sad-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sad
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
