# aidw-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME aidw
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
