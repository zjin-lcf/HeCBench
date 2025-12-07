# sa-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sa
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
