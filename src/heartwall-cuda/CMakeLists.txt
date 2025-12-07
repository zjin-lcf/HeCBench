# heartwall-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME heartwall
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
