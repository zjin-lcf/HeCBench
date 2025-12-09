# rotary-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rotary
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
