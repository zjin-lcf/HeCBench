# attention-paged-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME attention-paged
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
