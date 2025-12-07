# cooling-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cooling
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
