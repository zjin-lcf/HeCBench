# topk-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME topk
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
