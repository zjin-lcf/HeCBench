# meanshift-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME meanshift
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
