# warpsort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpsort
    MODEL cuda
    SOURCES main.cpp warpsort.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -DWARP_SIZE=32 -DHALF_WARP_SIZE=16
)
