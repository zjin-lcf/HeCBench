# adamw-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME adamw
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
