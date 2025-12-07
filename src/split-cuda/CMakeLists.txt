# split-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME split
    MODEL cuda
    SOURCES main.cu verify.cu
    CATEGORIES algorithms
)
