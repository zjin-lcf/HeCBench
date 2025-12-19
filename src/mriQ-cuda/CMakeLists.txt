# mriQ-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mriQ
    MODEL cuda
    SOURCES main.cu file.cc
    CATEGORIES algorithms
    COMPILE_OPTIONS -use_fast_math
)
