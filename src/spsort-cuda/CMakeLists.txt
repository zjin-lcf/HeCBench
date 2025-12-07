# spsort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spsort
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
