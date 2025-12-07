# scan3-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan3
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
