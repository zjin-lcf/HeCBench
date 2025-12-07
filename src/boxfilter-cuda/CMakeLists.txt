# boxfilter-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME boxfilter
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
