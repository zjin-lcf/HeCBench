# svd3x3-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME svd3x3
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
