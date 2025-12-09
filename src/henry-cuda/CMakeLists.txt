# henry-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME henry
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
