# attention-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME attention
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
