# mcpr-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcpr
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
