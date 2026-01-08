# adam-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME adam
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
