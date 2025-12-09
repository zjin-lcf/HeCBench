# haversine-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME haversine
    MODEL cuda
    SOURCES distance.cu main.cpp
    CATEGORIES algorithms
)
