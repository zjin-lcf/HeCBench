# hexciton-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hexciton
    MODEL omp
    SOURCES kernels.cpp main.cpp reference.cpp utils.cpp
    CATEGORIES algorithms
)
