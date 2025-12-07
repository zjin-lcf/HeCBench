# swish-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME swish
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
