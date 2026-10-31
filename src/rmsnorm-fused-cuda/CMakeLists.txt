# rmsnorm-fused-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm-fused
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
