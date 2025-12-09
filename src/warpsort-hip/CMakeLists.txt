# warpsort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpsort
    MODEL hip
    SOURCES main.cpp warpsort.cu
    CATEGORIES algorithms
)
