# lavaMD-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lavaMD
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
