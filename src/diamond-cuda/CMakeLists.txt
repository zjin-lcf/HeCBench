# diamond-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME diamond
    MODEL cuda
    SOURCES masking.cu
    CATEGORIES algorithms
)
