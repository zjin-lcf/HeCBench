# mergeVS-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mergeVS
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
