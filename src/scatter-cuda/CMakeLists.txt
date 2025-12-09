# scatter-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatter
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
