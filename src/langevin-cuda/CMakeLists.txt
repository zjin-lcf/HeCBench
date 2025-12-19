# langevin-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -use_fast_math
)
