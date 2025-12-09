# deredundancy-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL omp
    SOURCES kernels.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
