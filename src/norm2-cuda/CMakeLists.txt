# norm2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME norm2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
