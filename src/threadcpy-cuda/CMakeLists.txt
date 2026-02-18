# threadcpy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadcpy
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
