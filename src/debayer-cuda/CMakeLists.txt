# debayer-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME debayer
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
