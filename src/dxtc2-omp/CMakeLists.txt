# dxtc2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME dxtc2
    MODEL omp
    SOURCES block.cpp kernel.cpp main.cpp
    CATEGORIES algorithms
)
