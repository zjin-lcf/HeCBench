# hotspot-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hotspot
    MODEL cuda
    SOURCES hotspot.cu
    CATEGORIES algorithms
)
