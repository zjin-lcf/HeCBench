# asmooth-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME asmooth
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
