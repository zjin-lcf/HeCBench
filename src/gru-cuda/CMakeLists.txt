# gru-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gru
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
