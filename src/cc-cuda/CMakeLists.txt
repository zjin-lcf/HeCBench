# cc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cc
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
