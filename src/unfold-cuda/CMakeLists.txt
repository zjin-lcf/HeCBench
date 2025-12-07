# unfold-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME unfold
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
