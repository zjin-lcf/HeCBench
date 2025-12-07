# p4-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME p4
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
