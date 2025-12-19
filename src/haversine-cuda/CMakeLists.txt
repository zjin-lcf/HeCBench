# haversine-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME haversine
    MODEL cuda
    SOURCES distance.cu main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -use_fast_math
)
