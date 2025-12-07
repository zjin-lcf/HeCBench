# extend2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME extend2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
