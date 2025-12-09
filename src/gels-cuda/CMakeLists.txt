# gels-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gels
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
