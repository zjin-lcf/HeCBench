# qkv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME qkv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
