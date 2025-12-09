# sps2d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sps2d
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
