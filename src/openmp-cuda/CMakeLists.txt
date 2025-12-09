# openmp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME openmp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
