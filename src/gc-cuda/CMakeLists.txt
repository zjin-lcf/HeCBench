# gc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gc
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
