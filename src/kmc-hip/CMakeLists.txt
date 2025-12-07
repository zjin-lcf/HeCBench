# kmc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmc
    MODEL hip
    SOURCES main.cpp
    CATEGORIES algorithms
)
