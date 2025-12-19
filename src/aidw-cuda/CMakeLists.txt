# aidw-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME aidw
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -use_fast_math
)
